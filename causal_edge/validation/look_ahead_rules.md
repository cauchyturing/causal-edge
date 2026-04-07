# Look-Ahead Semantic Review Rules

When `causal-edge validate` outputs `SEMANTIC REVIEW: required`, apply these rules to the strategy source code. Report violations as structured findings.

## Rules

### R-SHIFT: Every feature must be shifted before use in decisions

Track through variable assignments:
```python
# VIOLATION: temp is not shifted before use in position
temp = ret.rolling(20).mean()
positions = np.where(temp > 0, 1, 0)

# CLEAN:
temp = ret.rolling(20).mean().shift(1)
positions = np.where(temp > 0, 1, 0)

# VIOLATION: indirect — shift is missing across assignment chain
a = ret.rolling(20).mean()
b = a * 2
positions = np.where(b > 0, 1, 0)  # b inherits a's lack of shift
```

### R-ROLLING: rolling() result includes current value

Any `rolling(N).stat()` result at index `i` includes data up to `i`. Must `.shift(1)` before using for decisions at time `i`.

Applies to: `.mean()`, `.std()`, `.sum()`, `.var()`, `.corr()`, `.median()`, `.min()`, `.max()`

```python
# VIOLATION:
vol = ret.rolling(20).std()
positions[vol < 0.02] = 0  # vol[i] includes ret[i]

# CLEAN:
vol = ret.rolling(20).std().shift(1)
```

### R-GLOBAL: No full-array statistics for features

`np.std(array)`, `pd.Series.std()`, `array.mean()` on the full array uses future data.

```python
# VIOLATION:
zscore = (ret - np.mean(ret)) / np.std(ret)  # mean/std of FULL series

# CLEAN:
zscore = (ret - ret.expanding().mean().shift(1)) / ret.expanding().std().shift(1)
```

Exceptions: metrics computation AFTER backtest (Sharpe, MaxDD) is fine.

### R-WF: Walk-forward must exclude current day

Training window `[:i]` means "up to but NOT including i". `[:i+1]` includes `i` = leak.

```python
# VIOLATION:
X_train = X[:i+1]  # includes today
model.fit(X_train, y[:i+1])

# CLEAN:
X_train = X[:i]
model.fit(X_train, y[:i])
```

### R-TREND: Trend filter uses yesterday

Price comparison for today's position must use yesterday's values.

```python
# VIOLATION:
if close[i] < sma[i]: positions[i] = 0  # uses today's close

# CLEAN:
if close[i-1] < sma[i-1]: positions[i] = 0
# or: if close.shift(1).iloc[i] < sma.shift(1).iloc[i]:
```

### R-CORR: Cross-correlation must double-shift

`parent.shift(lag).rolling(N).corr(target)` — the `.corr()` at index `i` includes `target[i]`. Need `.shift(1)` after corr.

```python
# VIOLATION:
xcorr = parent.shift(14).rolling(60).corr(target)
# xcorr[i] uses target[i]

# CLEAN:
xcorr = parent.shift(14).rolling(60).corr(target.shift(1)).shift(1)
```

### R-EXPANDING: expanding() includes current value

Same as rolling — `expanding().stat()` at index `i` includes `i`. Must shift.

```python
# VIOLATION:
median = series.expanding().median()
threshold = median  # includes current value

# CLEAN:
median = series.expanding().median().shift(1)
```

## How to Report

Output violations as a list:
```
SEMANTIC VIOLATION: R-SHIFT L42 — temp = rolling().mean() used without shift at L45
SEMANTIC VIOLATION: R-CORR L78 — xcorr missing .shift(1) after .corr()
SEMANTIC CLEAN: no violations found
```

If violations found: strategy should be marked as needing fix before deployment.
