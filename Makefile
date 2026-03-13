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

.PHONY: qmia-bench qmia-bench-smoke qmia-bench-sklearn qmia-bench-strong qmia-bench-summary qmia-bench-summary-sklearn qmia-bench-summary-strong

qmia-bench:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--dataset-source $(DATASET_SOURCE) \
		--sklearn-datasets $(SKLEARN_DATASETS) \
		--lira-shadow-models $(LIRA_SHADOW_MODELS) \
		--qmia-alpha $(QMIA_ALPHA) \
		--qmia-iterations $(QMIA_ITERATIONS) \
		--qmia-depth $(QMIA_DEPTH) \
		--qmia-learning-rate $(QMIA_LEARNING_RATE) \
		--qmia-l2-leaf-reg $(QMIA_L2_LEAF_REG) \
		--qmia-subsample $(QMIA_SUBSAMPLE) \
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
		--lira-shadow-models $(LIRA_SHADOW_MODELS) \
		--qmia-alpha $(QMIA_ALPHA) \
		--qmia-iterations $(QMIA_ITERATIONS) \
		--qmia-depth $(QMIA_DEPTH) \
		--qmia-learning-rate $(QMIA_LEARNING_RATE) \
		--qmia-l2-leaf-reg $(QMIA_L2_LEAF_REG) \
		--qmia-subsample $(QMIA_SUBSAMPLE) \
		--out-json outputs/benchmarks/qmia_vs_lira_sklearn_make.json \
		--out-csv outputs/benchmarks/qmia_vs_lira_sklearn_make.csv

qmia-bench-strong:
	$(PYTHON) $(QMIA_BENCH_SCRIPT) \
		--dataset-source $(DATASET_SOURCE) \
		--sklearn-datasets $(SKLEARN_DATASETS) \
		--lira-shadow-models 20,40,100 \
		--qmia-alpha 0.02 \
		--qmia-iterations 200 \
		--qmia-depth 6 \
		--qmia-learning-rate 0.03 \
		--qmia-l2-leaf-reg 5.0 \
		--qmia-subsample 0.9 \
		--out-json outputs/benchmarks/qmia_vs_lira_strong_make.json \
		--out-csv outputs/benchmarks/qmia_vs_lira_strong_make.csv

qmia-bench-summary:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) $(QMIA_BENCH_JSON)

qmia-bench-summary-sklearn:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) outputs/benchmarks/qmia_vs_lira_sklearn_make.json

qmia-bench-summary-strong:
	$(PYTHON) $(QMIA_SUMMARY_SCRIPT) outputs/benchmarks/qmia_vs_lira_strong_make.json
