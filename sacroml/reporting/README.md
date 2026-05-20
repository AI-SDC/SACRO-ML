# sacroml.reporting

The `sacroml.reporting` module is used to generate experiment reports from JSON inputs and a config file.

---

## Quick Start

### 1. Prepare inputs

You need:
- a JSON file containing experiment results
- a YAML config file

Example:

---

### 2. Minimal working input

#### `attack_report.json`

```json
{
  "report_schema_version": "1.0",
  "metric_catalog": {
    "metrics": {
      "accuracy": {
        "label": "Accuracy",
        "description": "Model accuracy",
        "units": null,
        "higher_is_better": true,
        "category": "performance",
        "typical_range": "0-1",
        "notes": "",
        "allowed_aggregations": ["mean", "min", "max"]
      }
    }
  },
  "parameter_catalog": {
    "parameters": {
      "epsilon": {
        "label": "Epsilon",
        "description": "Privacy budget"
      }
    }
  },
  "attack_category_catalog": {
    "categories": {
      "mia": {
        "label": "Membership Inference",
        "description": "Membership inference attacks",
        "order": 1
      }
    }
  },
  "attack_catalog": {
    "attacks": {
      "simple_attack": {
        "label": "Simple Attack",
        "description": "Example attack",
        "attack_category": "mia",
        "attack_params": {
          "epsilon": {
            "label": "Epsilon",
            "description": "Privacy parameter"
          }
        },
        "key_metrics": ["accuracy"]
      }
    }
  },
  "attacks": {
    "experiment_1": {
      "log_time": "2026-01-01 12:00",
      "metadata": {
        "attack_name": "simple_attack",
        "attack_params": {
          "epsilon": 1.0
        },
        "global_metrics": {
          "accuracy": 0.85
        }
      },
      "attack_experiment_logger": {
        "attack_instance_logger": {
          "instance_1": {
            "accuracy": 0.8
          },
          "instance_2": {
            "accuracy": 0.9
          }
        }
      }
    }
  }
}
```

#### `report.yaml`

```yaml

author: "Your Name"
project_name: "Example Privacy Report"
project_blurb: "This is a minimal example report."
recommendations: "Review model performance and privacy trade-offs."

```

### 3. Run a report

```python
import os
from sacroml.reporting.report import Report

report = Report(
    report_json="report_input/attack_report.json",
    report_yaml="report_input/report.yaml"
)

# Generate plots
report.render_visualisations("output")

# Render report
output = report.render_report(
    template_name="report.md.j2",
    output_dir="output",
)

# Save report
os.makedirs("output", exist_ok=True)
out_path = os.path.join("output", "report.md")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)

print(f"Report written to {out_path}")


```

### 4. Output

```
output/
├── report.md
└── figures/
    └── *.png

```