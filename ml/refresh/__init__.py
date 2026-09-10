"""refresh — AQUIS 6-hourly production refresh pipeline.

Two decoupled cadences (configurable in data/refresh/refresh_config.json):

  * FORECAST REFRESH  (default every 6 h)
      fetch GWL + driver/weather -> update local datasets -> run inference for
      every station (per-station anchor = latest observed GWL) -> 120 x 6h
      trajectory + 30-day endpoint -> publish forecast artifacts atomically.

  * MODEL UPDATE      (default daily)
      assemble newly available labelled observations -> full refit of the shared
      pooled trajectory XGBoost (incremental boosting is not safe for the current
      quantileerror/hist setup, so the update is a scheduled full refit) ->
      honest non-overlap backtest of the CANDIDATE against the incumbent on the
      SAME recent window -> promote ONLY if it passes the quality gates; on
      failure the previous production model/forecast remains untouched.

Everything that touches the filesystem goes through refresh/state.py (restart-safe
state + flock) and refresh/publish.py (staging + atomic rename + manifest commit),
so the running app can never observe a half-updated dataset.
"""