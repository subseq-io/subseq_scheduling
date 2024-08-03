use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result as AnyResult};
use uuid::Uuid;

use crate::bounds::{Constraints, Window};
use crate::event::{Connection, Event, EventId, PlanningPhase};
pub use crate::event::{Plan, PlannedEvent, Problem};
use crate::worker::{Capability, WorkerId, Worker};

pub struct ConnectionBlueprint(Uuid);

/// A blueprint for events that need to be planned.
pub struct EventBlueprint {
    pub id: Uuid,
    pub start: Option<f64>,
    pub duration: f64,
    /// Lower is higher priority
    pub priority: i64,
    pub assigned_worker: Option<WorkerId>,
    pub requirements: HashSet<Capability>,
    pub constraints: Constraints,
    pub dependencies: Vec<ConnectionBlueprint>,
    pub children: Vec<EventBlueprint>,
}

impl EventBlueprint {
    pub fn new(id: Uuid, duration: f64, priority: i64) -> Self {
        EventBlueprint {
            id,
            start: None,
            duration,
            priority,
            assigned_worker: None,
            requirements: HashSet::new(),
            constraints: Constraints::new(),
            dependencies: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn start(mut self, start: f64) -> Self {
        self.start = Some(start);
        self
    }

    pub fn assigned_worker(mut self, assigned_worker: WorkerId) -> Self {
        self.assigned_worker = Some(assigned_worker);
        self
    }

    pub fn requirements(mut self, requirements: HashSet<Capability>) -> Self {
        self.requirements = requirements;
        self
    }

    pub fn constraints(mut self, constraints: Constraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn dependency(mut self, event_id: Uuid) -> Self {
        self.dependencies.push(ConnectionBlueprint(event_id));
        self
    }

    pub fn child(mut self, child: EventBlueprint) -> Self {
        self.children.push(child);
        self
    }
}

pub struct PlanBlueprint {
    pub(crate) events: Vec<Event>,
    pub(crate) event_map: HashMap<EventId, Uuid>,
    pub(crate) events_seen: HashMap<Uuid, EventId>,
    pub(crate) seed: [u8; 32],
    pub(crate) workers: Vec<Worker>,
    pub(crate) worker_map: HashMap<WorkerId, Uuid>,
}

impl PlanBlueprint {
    pub fn new(seed: [u8; 32]) -> Self {
        PlanBlueprint {
            events: Vec::new(),
            event_map: HashMap::new(),
            events_seen: HashMap::new(),
            seed,
            workers: Vec::new(),
            worker_map: HashMap::new(),
        }
    }

    pub fn plan(self) -> Result<Plan, (Plan, Vec<Problem>)> {
        let planning = PlanningPhase::from(self);
        planning.plan()
    }

    pub fn event(mut self, plan: EventBlueprint) -> AnyResult<Self> {
        let id = EventId(self.events.len());

        if let Some(assigned_worker) = &plan.assigned_worker {
            let worker = match self.workers.get_mut(assigned_worker.0) {
                Some(worker) => worker,
                None => {
                    return Err(anyhow!("{:?} not found", assigned_worker));
                }
            };

            if let Some(start) = plan.start {
                if !worker.is_available_for(Window{start, end: start + plan.duration}) {
                    return Err(anyhow!(
                        "{:?} is not available at the given start time",
                        assigned_worker
                    ));
                }
            }
        }

        let mut dependencies = vec![];
        for connection in plan.dependencies {
            let event_id = match self.events_seen.get(&connection.0) {
                Some(event_id) => *event_id,
                None => {
                    return Err(anyhow!("{:?} not found", connection.0));
                }
            };
            dependencies.push(Connection(event_id));
        }

        let event = Event::new(
            id,
            plan.start.unwrap_or(0.0),
            plan.duration,
            plan.priority,
            plan.assigned_worker,
            plan.requirements,
            plan.constraints,
            dependencies,
            None,
        );
        self.event_map.insert(id, plan.id);
        self.events_seen.insert(plan.id, id);
        self.events.push(event);
        Ok(self)
    }

    pub fn worker(mut self, worker_id: Uuid, blocked_off: Constraints, capabilities: HashSet<Capability>) -> Self {
        let id = WorkerId(self.workers.len());
        let worker = Worker::new(
            id,
            blocked_off,
            capabilities,
        );
        self.worker_map.insert(id, worker_id);
        self.workers.push(worker);
        self
    }
}
