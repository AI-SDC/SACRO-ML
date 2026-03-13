PYTHON ?= .venv/bin/python

QMIA_BENCH_SCRIPT := examples/sklearn/benchmark_qmia_vs_lira.py
QMIA_SUMMARY_SCRIPT := examples/sklearn/summarize_qmia_lira_benchmark.py
QMIA_BENCH_JSON ?= outputs/benchmarks/qmia_vs_lira_make.json
QMIA_BENCH_CSV ?= outputs/benchmarks/qmia_vs_lira_make.csv
LIRA_SHADOW_MODELS ?= 20,40
QMIA_ALPHA ?= 0.01
QMIA_ITERATIONS ?= 20
QMIA_DEPTH ?= 3
QMIA_LEARNING_RATE ?= 0.05
QMIA_L2_LEAF_REG ?= 3.0
QMIA_SUBSAMPLE ?= 0.8
DATASET_SOURCE ?= synthetic
SKLEARN_DATASETS ?= breast_cancer,wine_binary
RF_ESTIMATORS ?= 50
LARGE_SCENARIOS_JSON ?= examples/sklearn/qmia_lira_scenarios.large.json
LARGE_LIRA_SHADOW_MODELS ?= 20,40,80
LARGE_QMIA_ITERATIONS ?= 300
LARGE_QMIA_DEPTH ?= 6
LARGE_QMIA_LEARNING_RATE ?= 0.03
LARGE_QMIA_L2_LEAF_REG ?= 5.0
LARGE_QMIA_SUBSAMPLE ?= 0.9
FULL_SUMMARY_TXT ?= outputs/benchmarks/qmia_vs_lira_full_summary_make.txt
BENCH_JSONS := outputs/benchmarks/qmia_vs_lira_make.json outputs/benchmarks/qmia_vs_lira_sklearn_make.json outputs/benchmarks/qmia_vs_lira_strong_make.json outputs/benchmarks/qmia_vs_lira_large_make.json
COMMON_QMIA_ARGS := --qmia-alpha $(QMIA_ALPHA) --qmia-iterations $(QMIA_ITERATIONS) --qmia-depth $(QMIA_DEPTH) --qmia-learning-rate $(QMIA_LEARNING_RATE) --qmia-l2-leaf-reg $(QMIA_L2_LEAF_REG) --qmia-subsample $(QMIA_SUBSAMPLE)
CLEAN_PATTERNS := outputs/benchmarks/qmia_vs_lira*_make.json outputs/benchmarks/qmia_vs_lira*_make.csv $(FULL_SUMMARY_TXT)
CLEAN_FILES := $(wildcard $(CLEAN_PATTERNS))

.DEFAULT_GOAL := help

.PHONY: help clean qmia-bench qmia-bench-smoke qmia-bench-sklearn qmia-bench-strong qmia-bench-large qmia-bench-all qmia-bench-full qmia-bench-summary qmia-bench-summary-sklearn qmia-bench-summary-strong qmia-bench-summary-large qmia-bench-summary-full

help:
	@echo "Run targets:"
	@echo "  make clean                    Remove generated benchmark JSON/CSV files"
	@echo "  make qmia-bench               Run default QMIA vs LiRA benchmark"
	@echo "  make qmia-bench-smoke         Run a quick smoke benchmark"
	@echo "  make qmia-bench-sklearn       Run benchmark on sklearn datasets"
	@echo "  make qmia-bench-strong        Run stronger benchmark configuration"
	@echo "  make qmia-bench-large         Run larger synthetic benchmark sweep"
	@echo "  make qmia-bench-all           Run default + sklearn + strong + large benchmarks"
	@echo ""
	@echo "Summary targets:"
	@echo "  make qmia-bench-summary       Summarize default benchmark JSON"
	@echo "  make qmia-bench-summary-sklearn  Summarize sklearn benchmark JSON"
	@echo "  make qmia-bench-summary-strong   Summarize strong benchmark JSON"
	@echo "  make qmia-bench-summary-large    Summarize large benchmark JSON"
	@echo "  make qmia-bench-summary-full     Combined summary and save to text report"
	@echo ""
	@echo "Combined convenience target:"
	@echo "  make qmia-bench-full          Run all benchmarks, then run full summary"

clean:
ifneq ($(strip $(CLEAN_FILES)),)
	@rm -f $(CLEAN_FILES)
	@echo "Removed benchmark artifacts."
else
	@:
endif

qmia-bench:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--dataset-source $(DATASET_SOURCE) \
		--sklearn-datasets $(SKLEARN_DATASETS) \
		--rf-estimators $(RF_ESTIMATORS) \
		--lira-shadow-models $(LIRA_SHADOW_MODELS) \
		$(COMMON_QMIA_ARGS) \
		--out-json $(QMIA_BENCH_JSON) \
		--out-csv $(QMIA_BENCH_CSV)

qmia-bench-smoke:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--lira-shadow-models 5 \
		--qmia-iterations 20 \
		--qmia-depth 3 \
		--out-json outputs/benchmarks/qmia_vs_lira_smoke_make.json \
		--out-csv outputs/benchmarks/qmia_vs_lira_smoke_make.csv

qmia-bench-sklearn:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--dataset-source sklearn \
		--sklearn-datasets $(SKLEARN_DATASETS) \
		--rf-estimators $(RF_ESTIMATORS) \
		--lira-shadow-models $(LIRA_SHADOW_MODELS) \
		$(COMMON_QMIA_ARGS) \
		--out-json outputs/benchmarks/qmia_vs_lira_sklearn_make.json \
		--out-csv outputs/benchmarks/qmia_vs_lira_sklearn_make.csv

qmia-bench-strong:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--dataset-source $(DATASET_SOURCE) \
		--sklearn-datasets $(SKLEARN_DATASETS) \
		--rf-estimators $(RF_ESTIMATORS) \
		--lira-shadow-models 20,40,100 \
		--qmia-alpha 0.02 \
		--qmia-iterations 200 \
		--qmia-depth 6 \
		--qmia-learning-rate 0.03 \
		--qmia-l2-leaf-reg 5.0 \
		--qmia-subsample 0.9 \
		--out-json outputs/benchmarks/qmia_vs_lira_strong_make.json \
		--out-csv outputs/benchmarks/qmia_vs_lira_strong_make.csv

qmia-bench-large:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--dataset-source synthetic \
		--scenarios-json $(LARGE_SCENARIOS_JSON) \
		--rf-estimators $(RF_ESTIMATORS) \
		--lira-shadow-models $(LARGE_LIRA_SHADOW_MODELS) \
		--qmia-alpha $(QMIA_ALPHA) \
		--qmia-iterations $(LARGE_QMIA_ITERATIONS) \
		--qmia-depth $(LARGE_QMIA_DEPTH) \
		--qmia-learning-rate $(LARGE_QMIA_LEARNING_RATE) \
		--qmia-l2-leaf-reg $(LARGE_QMIA_L2_LEAF_REG) \
		--qmia-subsample $(LARGE_QMIA_SUBSAMPLE) \
		--out-json outputs/benchmarks/qmia_vs_lira_large_make.json \
		--out-csv outputs/benchmarks/qmia_vs_lira_large_make.csv

qmia-bench-all: qmia-bench qmia-bench-sklearn qmia-bench-strong qmia-bench-large

qmia-bench-full: qmia-bench-all qmia-bench-summary-full

qmia-bench-summary:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) $(QMIA_BENCH_JSON)

qmia-bench-summary-sklearn:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) outputs/benchmarks/qmia_vs_lira_sklearn_make.json

qmia-bench-summary-strong:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) outputs/benchmarks/qmia_vs_lira_strong_make.json

qmia-bench-summary-large:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) outputs/benchmarks/qmia_vs_lira_large_make.json

qmia-bench-summary-full:
	@mkdir -p outputs/benchmarks
	@echo "Writing full benchmark summary to: $(FULL_SUMMARY_TXT)"
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) $(BENCH_JSONS) | tee $(FULL_SUMMARY_TXT)
