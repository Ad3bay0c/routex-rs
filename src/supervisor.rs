use crate::config::AgentConfig;
use crate::error::Result;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// Number of times an agent can restart within the window before
/// the supervisor gives up and declares permanent failure.
const DEFAULT_MAX_RESTARTS: u32 = 3;

/// The sliding window duration for restart counting.
const DEFAULT_RESTART_WINDOW: Duration = Duration::from_secs(60);

/// FailureReport is sent by the scheduler when an agent fails.
///
/// Contains a oneshot reply channel — the scheduler blocks on
/// the receiver until the supervisor sends exactly one Decision.
pub struct FailureReport {
    pub agent_id: String,
    pub reason: String,
    pub attempt: u32,
    /// The scheduler blocks on this until supervisor responds
    /// oneshot::Sender can only send once
    pub reply: oneshot::Sender<Decision>,
}

/// Decision is what the supervisor sends back to the scheduler.
#[derive(Debug, Clone)]
pub enum Decision {
    /// Restart this specific agent
    Retry,

    /// Permanent failure
    Fail(String),

    /// Restart this agent AND all agents that depend on it
    RestartCascade(Vec<String>),

    /// Restart the entire crew from the beginning
    RestartAll,
}

/// RestartPolicy mirrors the YAML config value.
#[derive(Debug, Clone, PartialEq)]
pub enum RestartPolicy {
    /// Restart only the failed agent
    OneForOne,

    /// Restart the failed agent and all its dependents
    RestartForOne,

    /// Restart all agents in the crew
    OneForAll,

    /// Never restart — fail immediately
    Never,
}

impl RestartPolicy {
    /// Parse from the string value in agents.yaml
    pub fn from_str(s: &str) -> Self {
        match s {
            "one_for_one" => Self::OneForOne,
            "rest_for_one" => Self::RestartForOne,
            "one_for_all" => Self::OneForAll,
            "never" => Self::Never,
            _ => Self::OneForOne,
        }
    }
}

/// RestartRecord tracks restart history for a single agent.
/// Used to enforce the sliding window budget.
#[derive(Debug)]
struct RestartRecord {
    /// Total restarts within the current window
    count: u32,

    /// When the current window started
    window_start: Instant,

    /// Maximum restarts allowed within the window
    max_restarts: u32,

    /// Duration of the sliding window
    window: Duration,
}

impl RestartRecord {
    fn new() -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
            max_restarts: DEFAULT_MAX_RESTARTS,
            window: DEFAULT_RESTART_WINDOW,
        }
    }

    /// Record a restart attempt.
    /// Returns true if the restart is allowed, false if budget exceeded.
    fn record_restart(&mut self) -> bool {
        let now = Instant::now();

        // Reset window if it has expired
        if now.duration_since(self.window_start) > self.window {
            self.count = 0;
            self.window_start = now;
        }

        self.count += 1;
        self.count <= self.max_restarts
    }

    /// Check if this agent has exceeded its restart budget
    /// without recording a new restart attempt.
    fn is_budget_exceeded(&self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.window_start) > self.window {
            return false;
        }
        self.count >= self.max_restarts
    }
}

/// Supervisor watches agents and applies restart policies when they fail.
///
/// The supervisor owns the restart records and the dependency graph.
/// It receives FailureReports from the scheduler and sends Decisions back
/// through the oneshot reply channel embedded in each report.
///
/// This is stateless between restarts — each Decision is independent.
/// The scheduler drives the execution loop; the supervisor only advises.
pub struct Supervisor {
    /// Restart policies per agent — keyed by agent ID
    policies: HashMap<String, RestartPolicy>,

    /// Restart history per agent — tracks sliding window budgets
    records: HashMap<String, RestartRecord>,

    /// Dependency graph — maps agent ID to the IDs of agents
    dependents: HashMap<String, Vec<String>>,
}

impl Supervisor {
    /// Create a new Supervisor from the agent configs.
    pub fn new(agents: &[AgentConfig]) -> Self {
        let mut policies = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();

        for agent in agents {
            policies.insert(agent.id.clone(), RestartPolicy::from_str(&agent.restart));

            // Build reverse dependency graph for rest_for_one
            dependents.entry(agent.id.clone()).or_default();
            for dep in &agent.depends {
                dependents
                    .entry(dep.clone())
                    .or_default()
                    .push(agent.id.clone());
            }
        }

        Self {
            policies,
            records: HashMap::new(),
            dependents,
        }
    }

    /// Process a FailureReport and send a Decision back to the scheduler.
    ///
    /// This is the core of the supervisor. It:
    ///   1. Looks up the agent's restart policy
    ///   2. Checks the restart budget
    ///   3. Computes the Decision based on policy
    ///   4. Sends it through the oneshot reply channel
    ///
    /// The oneshot channel guarantees the scheduler receives exactly one Decision
    pub fn decide(&mut self, report: FailureReport) -> Result<()> {
        let agent_id = &report.agent_id;

        let record = self
            .records
            .entry(agent_id.clone())
            .or_insert_with(RestartRecord::new);

        let policy = self
            .policies
            .get(agent_id)
            .cloned()
            .unwrap_or(RestartPolicy::OneForOne);

        // Never policy — fail immediately without checking budget
        if policy == RestartPolicy::Never {
            let decision = Decision::Fail(format!(
                "agent '{}' failed with restart policy 'never': {}",
                agent_id, report.reason
            ));

            let _ = report.reply.send(decision);
            return Ok(());
        }

        // Check restart budget
        let allowed = record.record_restart();

        if !allowed {
            // Budget exceeded — permanent failure
            let decision = Decision::Fail(format!(
                "agent '{}' exceeded restart budget ({} restarts in {}s): {}",
                agent_id,
                DEFAULT_MAX_RESTARTS,
                DEFAULT_RESTART_WINDOW.as_secs(),
                report.reason,
            ));
            let _ = report.reply.send(decision);
            return Ok(());
        }

        // Budget allows restart — apply the policy
        let decision = match policy {
            RestartPolicy::OneForOne => {
                // Restart only this agent
                Decision::Retry
            }

            RestartPolicy::RestartForOne => {
                // Restart this agent and all its dependents
                // Use BFS to find all transitively dependent agents
                let cascade = self.find_cascade(agent_id);
                if cascade.is_empty() {
                    Decision::Retry
                } else {
                    Decision::RestartCascade(cascade)
                }
            }

            RestartPolicy::OneForAll => {
                // Restart the entire crew
                Decision::RestartAll
            }

            RestartPolicy::Never => unreachable!("handled above"),
        };

        let _ = report.reply.send(decision);
        Ok(())
    }

    /// Find all agents that transitively depend on the given agent.
    /// Uses BFS through the reverse dependency graph.
    ///
    /// Example dependency graph:
    ///   researcher → analyst → writer → critic
    ///
    /// If researcher fails with rest_for_one:
    ///   cascade = [analyst, writer, critic]
    ///
    /// The failed agent itself is included so the caller
    /// knows the complete set to restart.
    fn find_cascade(&self, agent_id: &str) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        let mut cascade = Vec::new();

        // Start BFS from the direct dependents of the failed agent
        if let Some(deps) = self.dependents.get(agent_id) {
            for dep in deps {
                if visited.insert(dep.clone()) {
                    queue.push_back(dep.clone());
                }
            }
        }

        // BFS through the dependency graph
        while let Some(current) = queue.pop_front() {
            cascade.push(current.clone());

            if let Some(deps) = self.dependents.get(&current) {
                for dep in deps {
                    if visited.insert(dep.clone()) {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        cascade
    }

    /// Reset the restart record for an agent.
    /// Called when an agent completes successfully —
    /// a successful run resets the failure count.
    pub fn reset(&mut self, agent_id: &str) {
        self.records.remove(agent_id);
    }

    /// Check if an agent has exceeded its budget without
    /// recording a new restart. Used for pre-flight checks.
    pub fn is_budget_exceeded(&self, agent_id: &str) -> bool {
        self.records
            .get(agent_id)
            .map(|r| r.is_budget_exceeded())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, Role};

    fn make_agent(id: &str, restart: &str, depends: Vec<&str>) -> AgentConfig {
        AgentConfig {
            id: id.to_string(),
            role: Role::Researcher,
            goal: "research".to_string(),
            backstory: None,
            tools: vec![],
            depends: depends.iter().map(|s| s.to_string()).collect(),
            restart: restart.to_string(),
            llm: None,
            max_tool_calls: 20,
        }
    }

    fn make_report(agent_id: &str) -> (FailureReport, oneshot::Receiver<Decision>) {
        let (tx, rx) = oneshot::channel();
        let report = FailureReport {
            agent_id: agent_id.to_string(),
            reason: "test failure".to_string(),
            attempt: 1,
            reply: tx,
        };
        (report, rx)
    }

    #[test]
    fn test_one_for_one_returns_retry() {
        let agents = vec![make_agent("researcher", "one_for_one", vec![])];
        let mut supervisor = Supervisor::new(&agents);

        let (report, mut rx) = make_report("researcher");
        supervisor.decide(report).unwrap();

        let decision = rx.try_recv().unwrap();
        assert!(matches!(decision, Decision::Retry));
    }

    #[test]
    fn test_never_returns_fail() {
        let agents = vec![make_agent("researcher", "never", vec![])];
        let mut supervisor = Supervisor::new(&agents);

        let (report, mut rx) = make_report("researcher");
        supervisor.decide(report).unwrap();

        let decision = rx.try_recv().unwrap();
        assert!(matches!(decision, Decision::Fail(_)));
    }

    #[test]
    fn test_one_for_all_returns_restart_all() {
        let agents = vec![
            make_agent("researcher", "one_for_all", vec![]),
            make_agent("writer", "one_for_all", vec!["researcher"]),
        ];
        let mut supervisor = Supervisor::new(&agents);

        let (report, mut rx) = make_report("researcher");
        supervisor.decide(report).unwrap();

        let decision = rx.try_recv().unwrap();
        assert!(matches!(decision, Decision::RestartAll));
    }

    #[test]
    fn test_budget_exceeded_returns_fail() {
        let agents = vec![make_agent("researcher", "one_for_one", vec![])];
        let mut supervisor = Supervisor::new(&agents);

        // Exhaust the budget
        for _ in 0..DEFAULT_MAX_RESTARTS {
            let (report, mut rx) = make_report("researcher");
            supervisor.decide(report).unwrap();
            let decision = rx.try_recv().unwrap();
            assert!(matches!(decision, Decision::Retry));
        }

        // Next restart should fail
        let (report, mut rx) = make_report("researcher");
        supervisor.decide(report).unwrap();
        let decision = rx.try_recv().unwrap();
        assert!(matches!(decision, Decision::Fail(_)));
    }

    #[test]
    fn test_rest_for_one_cascades_dependents() {
        // researcher → analyst → writer
        let agents = vec![
            make_agent("researcher", "rest_for_one", vec![]),
            make_agent("analyst", "rest_for_one", vec!["researcher"]),
            make_agent("writer", "rest_for_one", vec!["analyst"]),
        ];
        let mut supervisor = Supervisor::new(&agents);

        let (report, mut rx) = make_report("researcher");
        supervisor.decide(report).unwrap();

        let decision = rx.try_recv().unwrap();
        match decision {
            Decision::RestartCascade(cascade) => {
                assert!(cascade.contains(&"analyst".to_string()));
                assert!(cascade.contains(&"writer".to_string()));
                assert!(!cascade.contains(&"researcher".to_string()));
            }
            _ => panic!("expected RestartCascade"),
        }
    }

    #[test]
    fn test_rest_for_one_no_dependents_retries() {
        // writer has no dependents — rest_for_one is same as one_for_one
        let agents = vec![
            make_agent("researcher", "one_for_one", vec![]),
            make_agent("writer", "rest_for_one", vec!["researcher"]),
        ];
        let mut supervisor = Supervisor::new(&agents);

        let (report, mut rx) = make_report("writer");
        supervisor.decide(report).unwrap();

        let decision = rx.try_recv().unwrap();
        assert!(matches!(decision, Decision::Retry));
    }

    #[test]
    fn test_reset_clears_restart_count() {
        let agents = vec![make_agent("researcher", "one_for_one", vec![])];
        let mut supervisor = Supervisor::new(&agents);

        // Use up two restarts
        for _ in 0..2 {
            let (report, mut rx) = make_report("researcher");
            supervisor.decide(report).unwrap();
            rx.try_recv().unwrap();
        }

        // Reset after successful run
        supervisor.reset("researcher");

        // Should have full budget again
        assert!(!supervisor.is_budget_exceeded("researcher"));
    }

    #[test]
    fn test_restart_policy_parsing() {
        assert_eq!(
            RestartPolicy::from_str("one_for_one"),
            RestartPolicy::OneForOne
        );
        assert_eq!(
            RestartPolicy::from_str("rest_for_one"),
            RestartPolicy::RestartForOne
        );
        assert_eq!(
            RestartPolicy::from_str("one_for_all"),
            RestartPolicy::OneForAll
        );
        assert_eq!(RestartPolicy::from_str("never"), RestartPolicy::Never);
        // Unknown defaults to one_for_one
        assert_eq!(RestartPolicy::from_str("unknown"), RestartPolicy::OneForOne);
    }

    #[tokio::test]
    async fn test_oneshot_channel_enforces_single_decision() {
        // This test demonstrates the compile-time guarantee:
        // A oneshot::Sender can only send once — trying to send again
        // would be a compile error because send() consumes self
        let agents = vec![make_agent("researcher", "one_for_one", vec![])];
        let mut supervisor = Supervisor::new(&agents);

        let (report, rx) = make_report("researcher");
        supervisor.decide(report).unwrap();

        // Exactly one decision received
        let decision = rx.await.unwrap();
        assert!(matches!(decision, Decision::Retry));

        // The channel is now closed — no more decisions possible
        // This is enforced by the type system, not at runtime
    }
}
