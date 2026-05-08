use std::error::Error;
use std::fs;
use std::path::PathBuf;
use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(author, version, about = "Velo Fleet Capacity Projector")]
struct Args {
    /// Path to benchmark JSON report
    #[arg(short, long)]
    input: PathBuf,

    /// Number of nodes in the fleet
    #[arg(short, long, default_value_t = 100)]
    nodes: usize,

    /// Electricity cost ($/kWh)
    #[arg(short, long, default_value_t = 0.12)]
    cost_kwh: f64,

    /// Daily uptime (hours)
    #[arg(short, long, default_value_t = 24.0)]
    uptime_hours: f64,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BenchmarkRow {
    model_name: String,
    backend_name: String,
    test: String,
    avg_ts: f64,
    avg_power_w: Option<f64>,
    tokens_per_joule: Option<f64>,
}

fn generate_report(args: &Args, content: &str) -> Result<String, Box<dyn Error>> {
    let rows: Vec<BenchmarkRow> = serde_json::from_str(content)?;
    use std::fmt::Write;
    let mut out = String::new();

    writeln!(out, "--- Velo Fleet Projection Report ---")?;
    writeln!(out, "Nodes: {}", args.nodes)?;
    writeln!(out, "Electricity Cost: ${}/kWh", args.cost_kwh)?;
    writeln!(out, "Daily Uptime: {} hours", args.uptime_hours)?;
    writeln!(out, "------------------------------------")?;

    for row in rows {
        let fleet_tps = row.avg_ts * args.nodes as f64;
        let daily_tokens = fleet_tps * 3600.0 * args.uptime_hours;
        
        let (fleet_kw, daily_cost) = if let Some(power_w) = row.avg_power_w {
            let fleet_kw = (power_w * args.nodes as f64) / 1000.0;
            let daily_kwh = fleet_kw * args.uptime_hours;
            let daily_cost = daily_kwh * args.cost_kwh;
            (Some(fleet_kw), Some(daily_cost))
        } else {
            (None, None)
        };

        let cost_per_m_tokens = if let Some(cost) = daily_cost {
            Some((cost / daily_tokens) * 1_000_000.0)
        } else {
            None
        };

        writeln!(out, "Test: {} | Model: {}", row.test, row.model_name)?;
        writeln!(out, "  Fleet Throughput:  {:.2} tokens/sec", fleet_tps)?;
        writeln!(out, "  Daily Capacity:    {:.2} billion tokens", daily_tokens / 1e9)?;
        
        if let Some(kw) = fleet_kw {
            writeln!(out, "  Fleet Power:       {:.2} kW", kw)?;
            writeln!(out, "  Daily Elec. Cost:  ${:.2}", daily_cost.unwrap())?;
            writeln!(out, "  Cost per 1M tok:   ${:.6}", cost_per_m_tokens.unwrap())?;
        } else {
            writeln!(out, "  Power Data:        N/A (run benchmark with --power)")?;
        }
        writeln!(out)?;
    }

    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let content = fs::read_to_string(&args.input)?;
    let report = generate_report(&args, &content)?;
    print!("{}", report);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_report() {
        let args = Args {
            input: PathBuf::from("test.json"),
            nodes: 100,
            cost_kwh: 0.12,
            uptime_hours: 24.0,
        };
        let content = r#"[
            {
                "model_name": "llama",
                "backend_name": "cpu",
                "test": "smoke",
                "avg_ts": 10.0,
                "avg_power_w": 150.0,
                "tokens_per_joule": null
            },
            {
                "model_name": "llama",
                "backend_name": "cpu",
                "test": "smoke",
                "avg_ts": 10.0,
                "avg_power_w": null,
                "tokens_per_joule": null
            }
        ]"#;
        
        let report = generate_report(&args, content).unwrap();
        assert!(report.contains("Nodes: 100"));
        assert!(report.contains("Fleet Power:       15.00 kW"));
        assert!(report.contains("Power Data:        N/A"));
    }
}
