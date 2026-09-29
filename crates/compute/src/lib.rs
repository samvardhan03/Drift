pub mod context;
pub mod cvar;
pub mod data;
pub mod dispatch;
pub mod drift;
pub mod error;
pub mod experiments;
pub mod format;
pub mod model;
pub mod performance;
pub mod policy;
pub mod portfolio;
pub mod regime;
pub mod reverse_stress;
pub mod scenarios;
pub mod trace;

pub use error::{ComputeError, Result};
pub use format::format_inr;
