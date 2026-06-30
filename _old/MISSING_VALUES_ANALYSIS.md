# Analysis of Missing Values in Synthetic Tracks Output

## Summary
Missing values in `final_wind_speed`, `V_0`, `t_L_steps`, and related variables in the intensified tracks output result from **four distinct mechanisms**:

1. **Skipped Loop Processing** (BUG?): Cyclones marked for stopping remain in output with NaN values ⚠️
2. **Initial Genesis (Step 0)**: Loop initialization step not processed
3. **Ocean Phase**: Pre-landfall evolution without land-decay state tracking
4. **Land Interactions**: Land-decay tracking only active when over land

**Most Critical**: Mechanism #1 appears to be a **bug** where cyclones are not being filtered correctly by the `keep_mask` logic, leaving unprocessed rows in the output.

---

## Detailed Analysis

### 0. IMPORTANT: Rows Skipped from Loop Processing (NEW)

**Pattern**: Cyclones that should be filtered out by the `keep_mask` logic remain in output with NaN values

**Root Cause** (Bug in `intensify.py`):
When a cyclone is marked for stopping via `stop_step` dict, subsequent loop iterations skip processing that cyclone. However, the `keep_mask` filtering at the end (lines 427-438) may not be removing these rows correctly.

**Evidence** from `/data/output/catherina/intensified_tracks/ACCESS-CM2/ssp245/seed=0/year=2091/month=8`:
- **SID 357140371**:
  - Step 0: `wind_speed=20.0` ✓ (initialized)
  - Step 1: `wind_speed=20.0` ✓ (processed successfully)
  - Steps 2-175: `wind_speed=NaN` ✗ (skipped from loop, never processed)
  - `is_on_land` transitions to True at step 16 with `wind_speed=NaN`
  - `t_L_steps` increments correctly (0.0, 1.0, 2.0...) but `V_0` remains NaN

**What happens**:
1. Cyclone is active and processed through step 1
2. At step 2, cyclone is no longer in `active_idx` (either already stopped or filtered)
3. Loop skips processing for that cyclone using: `if (seed, SID) not in stop_step: continue`
4. Rows 2+ remain in dataframe but are **never updated** - they retain initial NaN values
5. `keep_mask` should filter these rows but **doesn't** (possible bug)

**The Code Issue** (`src/intensitifcation_and_decay/intensify.py` lines 427-438):
```python
if stop_step:
    def _keep_row(index_tuple):
        seed = index_tuple[...]  # MultiIndex unpacking
        SID = index_tuple[...]
        step = index_tuple[...]
        stp = stop_step.get((seed, SID))
        return (stp is None) or (step <= stp-1)  # Logic: keep if not stopped OR before stop step
    
    keep_mask = tracks.index.map(_keep_row)
    tracks = tracks.loc[keep_mask].copy()  # Should delete rows with stp <= step
```

**Problem**: 
- The `keep_mask` filtering logic appears correct
- But cyclones marked for stopping still appear in output files with NaN values
- Suggests either: (a) `stop_step` is empty, (b) exception prevents filtering, or (c) rows re-added after filtering

**Why this produces cascading NaNs**:
- `final_wind_speed = NaN` (line 247: `tracks.loc[idx_i, "final_wind_speed"] = base_wind` not executed)
- All downstream calculations depend on shifted variables from step N-1
- When shifted from NaN row, produces NaN: `V_0_1`, `t_L_steps_1`, `qc`, `delta_Sm`, `X`, `MPI_hpa`

---

### 1. Missing Values at Step 0 (Initial Genesis Point)

**Pattern**: ALL rows at step 0 have NaN for:
- `final_wind_speed`
- `V_0` 
- `t_L_steps`
- `V_0_1`
- `t_L_steps_1`

**Root Cause**: 
In `src/intensitifcation_and_decay/intensify.py`, the `intensify_track_forward()` function processes cyclones step-by-step starting from **step 1**:

```python
for i in range(1, global_max_step+1):  # Loop starts at 1, NOT 0
```

Step 0 represents the **initial genesis point** where cyclones are placed synthetically. While some variables are initialized at step 0 in the `preprocessing_cyclone()` function:
- `wind_speed` is set to 20.0 m/s (initial wind at genesis)
- `delta_pc_hpa` is set to 0 (no pressure drop yet)
- `pc_hpa` is calculated from wind-pressure relationship

However, the **intensification simulation loop is skipped** for step 0, leaving:

```python
# Initialization (never updated for step 0):
tracks["V_0"] = np.nan              
tracks["t_L_steps"] = np.nan        
tracks["final_wind_speed"] = np.nan 
```

**Rationale**: Step 0 is a *boundary condition* representing where the cyclone begins, not a time-integrated state. The intensification model only runs forward in time starting from step 1.

---

### 2. Missing Values in `V_0` and `t_L_steps` for Most Cyclones

**Pattern**: 
- Most cyclones have `V_0 = NaN` and `t_L_steps = NaN` for their entire lifespans
- V_0 and t_L_steps only contain values when `is_on_land = True` (cyclone is over land)

**Example from data** (3 combined files, 851 total rows):
- `V_0` non-NaN: 20 out of 851 rows (~2.3%)
- `t_L_steps` non-NaN: 148 out of 851 rows (~17.4%)
- Breakdown by step:
  - Step 0-4: 0% have V_0 values (cyclones at sea, haven't made landfall)
  - Step 5+: Only rows with `is_on_land=True` have V_0 values

**Root Cause**:
`V_0` and `t_L_steps` are **land-decay state variables** that track cyclone behavior specifically when over land. They are set in the intensification loop under these conditions:

```python
# V_0 is only set when cyclone crosses from ocean to land (LANDFALL):
landfall_mask = is_land_now & (~was_land_prev)
if landfall_mask.any():
    tracks.loc[landfall_mask.index[landfall_mask], "V_0"] = base_wind.loc[landfall_mask]

# t_L_steps is only set while on land:
if landfall_mask.any():
    tracks.loc[landfall_mask.index[landfall_mask], "t_L_steps"] = 0
if cont_land_mask.any():  # Continuing on land
    tracks.loc[cont_land_mask.index[cont_land_mask], "t_L_steps"] = prev_counts + 1
if off_land_mask.any():  # Leaving land
    tracks.loc[off_land_mask.index[off_land_mask], "t_L_steps"] = np.nan
```

**Interpretation**: 
- **V_0** = The wind speed at which the cyclone made landfall (captures intensity at landfall)
- **t_L_steps** = How many consecutive timesteps the cyclone has spent on land (decaying over land)

Since many simulated cyclones:
1. Never make landfall (remain over ocean until decay)
2. Make landfall but are then removed from simulation (SST < 15°C, wind < 5 m/s, rolling avg < 10 m/s)

These variables remain `NaN` for the vast majority of cyclones.

---

### 3. Cascade of Missing Values in Related Variables

**Pattern**: When `final_wind_speed` is NaN at step 0, the following variables are also NaN:
- `V_0_1` (shifted V_0 from previous step)
- `t_L_steps_1` (shifted t_L_steps from previous step)  
- `delta_pc_hpa_1`, `pc_hpa_1`, `pc_hpa_1_eff` (all shifted variables)
- Downstream computed variables: `qc`, `delta_Sm`, `X`, `MPI_hpa`, `mpd_hpa`, `mslp_mpd_diff_hpa`

**Root Cause**:
All shifted variables use `.shift(1)` to retrieve values from the previous step:

```python
tracks["V_0_1"]          = tracks.groupby(level=['seed','SID'])["V_0"].shift(1)
tracks["t_L_steps_1"]    = tracks.groupby(level=['seed','SID'])["t_L_steps"].shift(1)
tracks["delta_pc_hpa_1"] = tracks.groupby(level=['seed','SID'])["delta_pc_hpa"].shift(1)
```

At step 0 (first step of each cyclone):
- There is no previous step, so `.shift(1)` returns `NaN`
- All downstream calculations that depend on shifted variables then propagate `NaN`

Even with `.fillna()` calls in some places, the key downstream variables (`qc`, `delta_Sm`, `X`) cannot be computed when their required inputs are NaN:

```python
tracks.loc[idx_i, "qc"] = qc(
    pc_shifted_hpa=tracks.loc[idx_i, "pc_hpa_1_eff"],  # NaN at step 0!
    sst_kelvin=step_i["SST"] + 273.15,
)
```

---

## Critical Finding: Data Quality Issue

**A subset of cyclones are NOT being processed correctly**, resulting in extensive NaN values:

### Example: SID 357140371 (year 2091, month 8)
- **Only step 1 has valid data**, all subsequent steps have `wind_speed=NaN`
- Cyclone eventually reaches land (step 16) but with `is_on_land=True` and `wind_speed=NaN`
- `t_L_steps` increment correctly (0.0 → 1.0 → 2.0...) but `V_0=NaN` throughout
- Stopping condition NOT met at step 1 (SST=31°C, wind=20 m/s), yet processing stops

**Implications**:
- These cyclones should have been **removed by `keep_mask` filtering** at lines 427-438
- Instead, they appear in output with unprocessed NaN values
- This is likely a **bug in the intensification code**, not expected behavior
- Affects unknown number of cyclones across all output files

---

### 3. Missing Values at Step 0 (Initial Genesis Point)

| Variable | Total Rows | Missing | % Missing | Location |
|----------|-----------|---------|-----------|----------|
| `final_wind_speed` | 295 | 4 | 1.4% | **All at step 0** |
| `V_0` | 295 | 295 | 100% | Only on land (rare) |
| `t_L_steps` | 295 | 295 | 100% | Only on land (rare) |
| `V_0_1` | 295 | 295 | 100% | Shifted from V_0 |
| `t_L_steps_1` | 295 | 295 | 100% | Shifted from t_L_steps |
| `delta_pc_hpa_1` | 295 | 4 | 1.4% | At step 0 (no previous) |
| `pc_hpa_1` | 295 | 4 | 1.4% | At step 0 (no previous) |
| `qc` | 295 | 4 | 1.4% | Computed from shifted vars |
| `delta_Sm` | 295 | 4 | 1.4% | Computed from shifted vars |
| `X` | 295 | 4 | 1.4% | Computed from shifted vars |
| `MPI_hpa` | 295 | 4 | 1.4% | Computed from shifted vars |

---

## Data Quality Summary

### File 1: `/data/output/catherina/intensified_tracks/ACCESS-CM2/ssp245/seed=0/year=2036/month=1/`
(All cyclones processed normally)

**Statistics**:
- Total rows: 295
- `final_wind_speed` missing: 4 (1.4%, all at step 0)
- `V_0` missing: 295 (100%, only on land when set)
- `t_L_steps` missing: 295 (100%, only on land when incremented)

### File 2: `/data/output/catherina/intensified_tracks/ACCESS-CM2/ssp245/seed=0/year=2091/month=8/`
(Contains unprocessed cyclones - QUALITY ISSUE)

**Detected issue**:
- SID 357140371: Only step 1 valid, steps 2-175 have `wind_speed=NaN`
- These cyclones should have been filtered by `keep_mask` but are not
- Unknown prevalence across all output files

---

## Why Most Missing Values Occur

The missing values reflect **intentional model design**:

1. **Step 0 Exclusion**: Loop starts at step 1, so step 0 is not processed (starting point only)

2. **Land-Decay Tracking**: `V_0` and `t_L_steps` only set when cyclones are on land. Most cyclones never make landfall → remain NaN

3. **Shifted Variables**: At step 0, there's no "previous step" → all shifted variables are NaN

**However**, mechanism #1 (unprocessed cyclones) is a **BUG**, not intended design.

---

## Recommendations for Users

### For Analysis:
1. **Filter out problematic rows**: Use `df[df['final_wind_speed'].notna() | (df.index.get_level_values('step') == 0)]`
2. **Detect unprocessed cycles**: Flag rows where `(step > 0) & (final_wind_speed.isna()) & (is_on_land==True)`
3. **Ocean-phase only**: Filter to `step > 0` when analyzing intensification
4. **Land-phase only**: Filter to `is_on_land == True` to study land decay

### For Code Developers:
1. **Debug `keep_mask` in intensify.py** (lines 427-438):
   - Add logging to track which cyclones enter `stop_step`
   - Verify MultiIndex unpacking works correctly
   - Test if exception is silently caught during row filtering

2. **Investigate early stopping**: SID 357140371 stops at step 2 with no clear stopping condition met

3. **Add data validation**: Post-processing check to detect step > 0 rows with `final_wind_speed=NaN`

### Example Code Filters:
```python
# Valid data only (exclude unprocessed):
valid = df[df['final_wind_speed'].notna() | (df['step'] == 0)]

# Quality check: find unprocessed:
unprocessed = df[(df['step'] > 0) & (df['final_wind_speed'].isna()) & (df['is_on_land']==True)]
print(f"Found {len(unprocessed)} unprocessed rows")

# Ocean phase:
ocean = df[df['step'] > 0]

# Land phase:
land = df[df['is_on_land'] == True]
```

---

## Code References & Known Issues

**Main Code**: `src/intensitifcation_and_decay/intensify.py`
- Line 225: `stop_step = {}` dictionary initialization
- Lines 239-242: Loop skipping for stopped cyclones
- Lines 420-425: Stop condition evaluation
- **Lines 427-438**: SUSPECTED BUG LOCATION - `keep_mask` filtering not removing unprocessed rows

**Why This Design?**

1. **Step 0 Exclusion**: Genesis locations are known, but their intensification history is unknown. Step 0 is the starting point before integration begins.

2. **Land-Decay Tracking**: The model only tracks land-specific state (`V_0`, `t_L_steps`) while cyclones are over land. Over ocean, these variables are not needed.

3. **Shifted Variables**: At boundary steps (step 0), there is no "previous step" for shifted variables.

**BUT**: Cyclones that should be deleted by `keep_mask` are still appearing in output files with NaN values - this is a **data integrity issue** that should be investigated.

---