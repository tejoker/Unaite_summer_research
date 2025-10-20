#!/usr/bin/env python3
"""
Answer key questions about spike detection behavior
"""

def question_2_why_windows_91_186_small_changes():
    """
    Q2: Why do windows 91-186 show SMALL changes when they contain the spike?
    """
    print("="*80)
    print("Q2: WHY DO WINDOWS 91-186 SHOW SMALL CHANGES?")
    print("="*80)

    print("""
HYPOTHESIS 1: The spike affects BOTH datasets similarly in those windows
-----------------------------------------------------------------------

Remember: You're comparing Golden vs Spike datasets.

Window 100 (contains spike at position ~90):
  - In GOLDEN dataset: Normal pattern throughout [110, 210)
  - In SPIKE dataset:  Normal pattern in [110, 199], spike at 200, normal in [201, 210)

  Result:
    - 90% of data is identical (rows 110-199)
    - 1% is spike (row 200)
    - 9% is normal after spike (rows 201-210)

    DynoTEARS optimization on 100 rows:
      - Golden: Learns from 100 normal rows
      - Spike:  Learns from 90 normal + 1 spike + 9 normal = mostly normal!

    -> Both learn SIMILAR causal structures (dominated by normal data)
    -> Small weight differences!


Window 187 (spike at position 3 - VERY START):
  - In GOLDEN dataset: Normal pattern throughout [197, 297)
  - In SPIKE dataset:  Normal in [197-199], SPIKE at 200, normal in [201-297)

  Result:
    - 3% normal before spike
    - 1% is spike
    - 96% normal AFTER spike

    The spike is RIGHT AT THE START!

    DynoTEARS tries to:
      - Explain 96 normal rows using relationships learned from 3 pre-spike rows
      - The spike DISRUPTS the initial conditions
      - Massive weight adjustments needed!

    -> Golden and Spike learn VERY DIFFERENT structures
    -> HUGE weight differences!


MATHEMATICAL EXPLANATION:
------------------------

DynoTEARS minimizes: ||Y - f(X, W)||² + λ|W|

Window with spike in MIDDLE (window 140):
  - Spike affects 1 row out of 100
  - 99 rows pull weights toward "normal" solution
  - 1 row pulls weights toward "abnormal" solution
  - Result: 99:1 ratio, normal dominates
  - Both Golden and Spike converge to similar W

Window with spike at START (window 187):
  - Spike affects initial conditions
  - All 96 subsequent rows must be explained using spike-corrupted start
  - The causal propagation is disrupted throughout
  - Result: Entire window's dynamics are affected
  - Golden and Spike converge to VERY different W


ANALOGY:
--------

Think of a video:
  - Spike in middle of video = one corrupted frame
    -> Most frames are clean, compression works normally

  - Spike at start of video = first keyframe is corrupted
    -> All subsequent frames (delta-encoded from keyframe) are affected
    -> Massive reconstruction error!

The spike at the START of the window has MAXIMUM IMPACT.
    """)


def question_3_learning_spike_as_pattern():
    """
    Q3: Why is it a problem if DynoTEARS learns the spike as part of the pattern?
    """
    print("\n" + "="*80)
    print("Q3: WHY IS IT A PROBLEM IF DYNOTEARS LEARNS SPIKE AS PATTERN?")
    print("="*80)

    print("""
SHORT ANSWER: It's NOT a problem! This is exactly what you want!
-----------------------------------------------------------------

Your confusion comes from thinking about this wrong. Let me clarify:

WHAT YOU'RE DOING:
------------------
You're comparing two DIFFERENT datasets:
  - Golden dataset (no anomaly)
  - Spike dataset (with anomaly)

Process:
  1. Run DynoTEARS on Golden -> Get W_golden
  2. Run DynoTEARS on Spike -> Get W_spike
  3. Compare W_golden vs W_spike -> Find differences

WHAT HAPPENS IN EACH WINDOW:
-----------------------------

Window 100 (spike in middle):
  Golden learns: "Normal causal pattern"
  Spike learns:  "Normal causal pattern" (because 99/100 rows are normal)
  Difference:    SMALL (both learned similar patterns)

Window 187 (spike at start):
  Golden learns: "Normal causal pattern"
  Spike learns:  "Disrupted causal pattern" (spike corrupts propagation)
  Difference:    LARGE (learned very different patterns)


WHY THIS IS GOOD:
-----------------

You WANT DynoTEARS to learn the spike as part of the pattern!

If DynoTEARS IGNORED the spike:
  - Spike data: Row 200 = 304.0
  - Model learned without spike
  - Prediction: 254.0
  - Residual: 50.0 (spike is just "noise")
  - Causal structure unchanged
  - NO WEIGHT DIFFERENCES!
  - Can't detect anomaly!

If DynoTEARS LEARNS the spike:
  - Spike data: Row 200 = 304.0
  - Model adapts to spike
  - Prediction: 304.0
  - Residual: 0.0
  - Causal structure CHANGED to accommodate spike
  - LARGE WEIGHT DIFFERENCES!
  - CAN detect anomaly!


THE KEY INSIGHT:
----------------

Anomaly detection via causal discovery detects:
  ✓ Changes in CAUSAL STRUCTURE
  ✗ NOT just outliers in the data

The spike at row 200 causes:
  - Window 187-188: Spike is at START -> changes causal structure
  - Windows 91-186: Spike is in MIDDLE -> doesn't change causal structure much

This is GOOD because you're detecting:
  "Where did the causal relationships change?"

NOT:
  "Where is there a big value in the data?"


COMPARISON TO OTHER METHODS:
-----------------------------

Statistical anomaly detection:
  - Would flag row 200 directly (Z-score = 22.1)
  - Precise: Points to exact row
  - Doesn't explain WHY it's anomalous

Causal anomaly detection (your method):
  - Flags windows 187-188 (contains row 200)
  - Less precise: ~100 row range
  - Explains WHAT CHANGED: "Temperatur Exzenterlager rechts relationship changed 642x"
  - More robust: Detects structural changes, not just outliers


YOU GET BOTH:
-------------

Use your weight-based detection to:
  1. Flag suspicious windows (187-188)
  2. Identify WHICH causal relationships changed
  3. Then do fine-grained search in that window range
  4. Find the exact row (200) using the sensor that changed

This is MORE POWERFUL than just flagging row 200!
    """)


def question_4_anomaly_position():
    """
    Q4: Can we return anomaly position in time series, not just window index?
    """
    print("\n" + "="*80)
    print("Q4: CAN WE RETURN ANOMALY POSITION IN TIME SERIES?")
    print("="*80)

    print("""
YES! Here's the complete pipeline:
-----------------------------------

STEP 1: Window-level detection (what you have now)
  Input:  Weight matrices W_golden, W_spike
  Output: Anomalous windows [187, 188]

STEP 2: Map windows to time ranges
  Window 187: rows [197, 297)
  Window 188: rows [198, 298)
  Union: rows [197, 298)

STEP 3: Identify affected sensors
  From weight analysis:
    - Temperatur Exzenterlager rechts_diff: -642x change
    - This is the PRIMARY affected sensor

STEP 4: Fine-grained localization
  Load time series for "Temperatur Exzenterlager rechts"
  Look at rows [197, 298)
  Find maximum deviation from expected

  Expected (from Golden): ~254.0
  Actual (in Spike): Row 200 = 304.0
  Deviation: 50.0

  Result: Anomaly at row 200


PSEUDOCODE:
-----------

def detect_and_localize_anomaly(golden_data, test_data):
    # Step 1: Get weight matrices
    W_golden = dynotears(golden_data)
    W_test = dynotears(test_data)

    # Step 2: Find anomalous windows
    anomalous_windows = compare_weights(W_golden, W_test, threshold=1.0)
    # Returns: [187, 188]

    # Step 3: Map to time range
    time_ranges = [window_to_timerange(w) for w in anomalous_windows]
    # Returns: [[197,297), [198,298)]

    # Step 4: Get union
    search_range = (min(r[0] for r in time_ranges),
                    max(r[1] for r in time_ranges))
    # Returns: (197, 298)

    # Step 5: Identify affected sensors
    affected_sensors = get_changed_sensors(W_golden, W_test, anomalous_windows)
    # Returns: ["Temperatur Exzenterlager rechts", ...]

    # Step 6: Fine-grained search
    for sensor in affected_sensors:
        golden_values = golden_data[sensor][search_range[0]:search_range[1]]
        test_values = test_data[sensor][search_range[0]:search_range[1]]

        diffs = abs(test_values - golden_values)
        anomaly_position = search_range[0] + argmax(diffs)

        if diffs.max() > threshold:
            return {
                'position': anomaly_position,
                'sensor': sensor,
                'expected_value': golden_values[anomaly_position - search_range[0]],
                'actual_value': test_values[anomaly_position - search_range[0]],
                'deviation': diffs.max(),
                'detected_windows': anomalous_windows,
                'search_range': search_range
            }


ACCURACY:
---------

Your method would return:
  - Detected windows: [187, 188]
  - Search range: [197, 298) (101 rows)
  - Affected sensor: "Temperatur Exzenterlager rechts"
  - Exact position: 200
  - Expected: 254.0, Actual: 304.0, Deviation: 50.0

Precision: 101 rows → 1 row = 1% precision
Error: 0 rows (exact!)


BENEFITS OVER SIMPLE THRESHOLD:
--------------------------------

Simple threshold method:
  - Flags row 200 immediately
  - Precision: 1 row
  - Information: "This value is too high"

Your causal method:
  - Flags rows [197, 298)
  - Precision: 101 rows → 1 row (after refinement)
  - Information:
    * "Causal structure changed in this region"
    * "Temperatur Exzenterlager rechts relationship disrupted"
    * "642x weight change in autoregressive connection"
    * "This suggests sudden external shock"

Much richer diagnostic information!
    """)


def main():
    question_2_why_windows_91_186_small_changes()
    question_3_learning_spike_as_pattern()
    question_4_anomaly_position()

    print("\n" + "="*80)
    print("SUMMARY: ANSWERS TO YOUR QUESTIONS")
    print("="*80)
    print("""
Q1: Can you flag window 188 and identify spike at t=200?
A1: YES, but requires refinement step
    - Flag windows 187-188 (coarse detection)
    - Map to time range [197, 298)
    - Analyze affected sensors in this range
    - Pinpoint exact row 200

Q2: Why do windows 91-186 show small changes?
A2: Spike in MIDDLE of window = dominated by 99% normal data
    Spike at START of window (187-188) = disrupts entire propagation
    Position in window matters MORE than presence!

Q3: Is learning spike as pattern a problem?
A3: NO! It's a FEATURE!
    - You WANT DynoTEARS to adapt to spike
    - This causes weight changes
    - Weight changes are what you detect
    - If it didn't adapt, no detection possible!

Q4: Can you return position in time series?
A4: YES! Full pipeline:
    - Window detection (coarse: ~100 rows)
    - Sensor identification (which relationships changed)
    - Fine-grained search (precise: exact row)
    - Result: row 200 with full diagnostic info
    """)

if __name__ == "__main__":
    main()
