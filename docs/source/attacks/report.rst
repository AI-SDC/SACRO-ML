Report
======

.. automodule:: sacroml.attacks.report
    :members:

Converting legacy reports
-------------------------

Older SACRO-ML versions write a flat ``report.json`` keyed by
``"<attack_name>_<log_id>"``. The new reporting pipeline expects a nested,
catalog-enriched document validated by the bundled JSON schema
(``sacroml/reporting/sacroml_attack_report.schema.json``).

Use the ``convert-report`` command to upgrade an existing report in place
without re-running any attacks:

.. prompt:: bash

    sacroml convert-report report.json report_new.json

The converter:

* wraps the legacy experiments under a top-level ``attacks`` key;
* normalises experiment metadata so it satisfies the schema (injecting a
  placeholder ``sacroml_version`` for very old reports, ensuring an
  ``attack_instance_logger`` exists for instance-less structural attacks,
  serialising non-scalar ``global_metrics`` values, etc.);
* injects the four human-readable catalogs (``metric_catalog``,
  ``parameter_catalog``, ``attack_category_catalog``, ``attack_catalog``)
  from the bundled common definitions in
  ``sacroml/reporting/catalog_definitions.json``;
* **warns** when a metric, parameter, attack or attack category observed in
  the report is not present in the catalogs (conversion still succeeds); and
* validates the result against the JSON schema.

Curve-valued arrays (``fpr`` / ``tpr`` / ``roc_thresh``) are passed through
unchanged. ``roc_thresh`` legitimately starts with ``null``, which the schema
does not yet permit, so such violations are reported as notices rather than
errors. Pass ``--no-validate`` to skip schema validation.

To extend the catalogs with site-specific metrics, parameters or attacks,
edit ``sacroml/reporting/catalog_definitions.json``.

.. automodule:: sacroml.reporting.convert
    :members: convert_report, convert_report_file, ConversionResult
