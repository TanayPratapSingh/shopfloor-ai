"""Manufacturing SOPs for milling machine operations — ingested into ChromaDB for RAG."""


def get_sop_documents() -> list[dict]:
    return [
        {
            "title": "Milling Machine Predictive Maintenance Protocol",
            "category": "Maintenance",
            "content": """
PREDICTIVE MAINTENANCE — MILLING OPERATIONS

1. OVERVIEW
This procedure covers sensor-based condition monitoring for CNC milling machines
running mixed-quality product variants (Low, Medium, High).

2. MONITORED PARAMETERS
  - Air temperature [K]: Ambient conditions. Normal range 295-305K.
    Sustained readings above 303K indicate cooling system degradation.
  - Process temperature [K]: Cutting zone temperature. Should track
    air temp + 8-12K. Delta above 12K signals excessive friction or
    insufficient coolant flow.
  - Rotational speed [rpm]: Spindle speed. Derived from a base power
    of 2860W with normal noise. Sudden drops indicate bearing wear
    or belt slippage.
  - Torque [Nm]: Cutting force. Normal mean ~40Nm (std ~10).
    Torque spikes above 60Nm suggest tool dulling or material anomaly.
  - Tool wear [min]: Cumulative minutes of active cutting.
    Replace at 200-240 min depending on product variant.

3. FAILURE MODE REFERENCE
  - TWF (Tool Wear Failure): Tool wear reaches 200-240 min threshold.
    Random chance of failure between these values.
    Prevention: Track wear, replace proactively at 190 min.
  - HDF (Heat Dissipation Failure): When difference between air and
    process temp falls below 8.6K AND rotational speed < 1380 rpm.
    Prevention: Monitor temp differential, verify coolant system.
  - PWF (Power Failure): Power = torque × (2π × speed / 60).
    Fails if power < 3500W or > 9000W.
    Prevention: Monitor power output, calibrate regularly.
  - OSF (Overstrain Failure): product of tool wear × torque exceeds
    threshold (11000 for L, 12000 for M, 13000 for H variants).
    Prevention: Reduce feed rate as tool wear increases.
  - RNF (Random Failure): 0.1% probability per cycle regardless of
    parameters. Irreducible — plan for it with buffer capacity.

4. ESCALATION MATRIX
  - Single failure event → Log, replace tool, resume production.
  - 2+ failures same mode in 8 hours → Maintenance supervisor review.
  - 3+ failures any mode in shift → Line stop, root cause analysis.
  - HDF or PWF → Immediate coolant system / electrical inspection.
""",
        },
        {
            "title": "Tool Wear Management and Replacement Schedule",
            "category": "Quality",
            "content": """
TOOL WEAR MANAGEMENT

1. WEAR TRACKING
  - Tool wear is measured in cumulative cutting minutes.
  - Wear rate varies by product quality variant:
    * High quality (H): Tighter tolerances, 5 min added wear per cycle.
    * Medium quality (M): Standard tolerances, 3 min per cycle.
    * Low quality (L): Loose tolerances, 2 min per cycle.

2. REPLACEMENT THRESHOLDS
  - Mandatory replacement: 240 min (absolute maximum, any variant).
  - Recommended replacement: 200 min (proactive, reduces TWF risk).
  - For High-quality products: replace at 190 min (precision required).
  - Tool wear between 200-240 min: random failure probability increases.

3. TOOL CHANGE PROCEDURE
  a. Pause production cycle (do NOT interrupt mid-cut).
  b. Retract spindle to safe position.
  c. Remove worn tool, inspect for chipping or abnormal wear patterns.
  d. Log wear reading and any observations in MES.
  e. Install fresh tool, verify torque on collet to spec.
  f. Run test cut on scrap piece, verify dimensional accuracy.
  g. Resume production, reset tool wear counter in system.

4. WEAR PATTERN ANALYSIS
  - Linear wear progression: Normal, expected behavior.
  - Accelerating wear (exponential): Material harder than expected
    or coolant insufficient. Flag for quality review.
  - Sudden spike: Likely tool chipping, inspect immediately.
  - Track wear vs. torque correlation — rising torque at constant
    speed indicates tool dulling before wear counter reflects it.

5. OVERSTRAIN PREVENTION
  - Overstrain threshold = Tool wear × Torque.
  - L variant: limit is 11,000 min·Nm
  - M variant: limit is 12,000 min·Nm
  - H variant: limit is 13,000 min·Nm
  - As tool wear increases, reduce feed rate to lower torque.
""",
        },
        {
            "title": "Temperature Monitoring and Heat Dissipation",
            "category": "Operations",
            "content": """
TEMPERATURE MONITORING — MILLING OPERATIONS

1. SENSOR CONFIGURATION
  - Air temperature sensor: Mounted 1m from machine, ambient reading.
    Normal: ~300K (27°C), std deviation 2K.
  - Process temperature sensor: Mounted near cutting zone.
    Normal: air temp + 10K, std deviation 1K.

2. HEAT DISSIPATION FAILURE (HDF) CRITERIA
  The machine fails when BOTH conditions are true simultaneously:
  - Temperature difference (process - air) < 8.6K
  - Rotational speed < 1380 rpm

  Root cause: Insufficient heat removal from cutting zone.
  When speed is low AND temp differential is small, heat builds
  in the workpiece and tool, causing thermal expansion and
  dimensional inaccuracy, leading to part rejection or tool fracture.

3. COOLANT SYSTEM CHECKS
  - Verify coolant flow rate every 4 hours.
  - Check coolant concentration weekly (target 5-8%).
  - Replace coolant every 30 days or when pH drops below 8.5.
  - Inspect nozzle alignment — coolant must hit the cutting zone
    directly, not spray into air.

4. AMBIENT TEMPERATURE MANAGEMENT
  - Shop floor HVAC target: 295-303K (22-30°C).
  - Readings above 303K for >30 min: alert maintenance.
  - Summer months: increase coolant concentration to 8%.
  - Winter months: allow 15-min warm-up cycle before production.

5. TEMPERATURE TREND ANALYSIS
  - Gradual rise in process temp at constant speed: tool dulling
    (increased friction generates more heat).
  - Sudden process temp drop: coolant flow increased or tool broke
    (no cutting friction).
  - Air temp fluctuation >4K within 1 hour: HVAC issue, log it.
""",
        },
        {
            "title": "Power and Torque Monitoring",
            "category": "Operations",
            "content": """
POWER AND TORQUE MONITORING

1. POWER CALCULATION
  Power (W) = Torque (Nm) × Angular velocity (rad/s)
  Angular velocity = 2π × Rotational speed (rpm) / 60

2. POWER FAILURE (PWF) CRITERIA
  Machine failure occurs when calculated power falls outside the
  safe operating envelope:
  - Power < 3,500W → Underpowered: insufficient cutting force,
    tool may deflect, causing poor surface finish or breakage.
  - Power > 9,000W → Overpowered: excessive force on spindle
    bearings, risk of mechanical failure.

3. TORQUE SPECIFICATIONS
  - Normal operating range: 20-60 Nm (mean ~40Nm).
  - Torque > 55 Nm sustained: monitor closely, reduce feed rate.
  - Torque > 70 Nm spike: emergency stop, inspect tool and workpiece.
  - Torque readings are product-variant dependent:
    * L variant: higher torque acceptable (less precision needed).
    * H variant: keep torque < 50 Nm for surface finish quality.

4. ROTATIONAL SPEED
  - Base calculation: speed = 2860W / (torque × 2π/60).
  - Normal range: 1200-2600 rpm for standard operations.
  - Below 1380 rpm combined with low temp differential → HDF risk.
  - Speed fluctuations > 200 rpm between consecutive reads: check
    belt tension and spindle bearing condition.

5. MONITORING PROTOCOL
  - Log power, torque, speed every cycle in MES.
  - Dashboard should display real-time power band (3500-9000W).
  - Automated alert when power exits safe envelope for > 3 cycles.
  - Weekly trend analysis: power drift indicates drive belt wear or
    spindle degradation before catastrophic failure.
""",
        },
        {
            "title": "Product Quality Variant Specifications",
            "category": "Quality",
            "content": """
PRODUCT QUALITY VARIANT SPECIFICATIONS

1. VARIANT DISTRIBUTION
  - L (Low quality): 50% of production volume.
    Wider tolerances, standard surface finish.
    Serial numbers: L50001 - L59999
  - M (Medium quality): 30% of production volume.
    Standard tolerances, good surface finish.
    Serial numbers: M20001 - M29999
  - H (High quality): 20% of production volume.
    Tight tolerances, fine surface finish required.
    Serial numbers: H10001 - H19999

2. VARIANT-SPECIFIC PARAMETERS
                        L           M           H
  Tolerance band:       ±0.1mm      ±0.05mm     ±0.02mm
  Surface finish Ra:    3.2μm       1.6μm       0.8μm
  Tool wear limit:      240 min     220 min     200 min
  Overstrain limit:     11,000      12,000      13,000
  Inspection:           Sampling    Every 10th  Every unit

3. PRODUCTION SCHEDULING
  - Run H variants first in shift (fresh tooling, stable temp).
  - L variants can run on aging tools (wider tolerance absorbs wear).
  - Never switch from L to H without tool change — accumulated
    wear from L runs may exceed H variant limits.

4. QUALITY CONTROL
  - H variants: 100% dimensional inspection + surface roughness.
  - M variants: 10% sampling, increase to 100% if 2+ rejects.
  - L variants: 5% sampling, visual inspection sufficient.
  - Any variant failing on machine failure → scrap, do not rework.

5. TRACEABILITY
  - Every unit tracked by Product ID (variant letter + serial).
  - Failure events linked to: Product ID, sensor readings at time
    of failure, tool wear state, operator ID.
  - Monthly Pareto analysis: which variant has highest failure rate.
""",
        },
        {
            "title": "Shift Operations and KPI Definitions",
            "category": "KPI",
            "content": """
SHIFT OPERATIONS AND KPI DEFINITIONS

1. SHIFT SCHEDULE
  - Day shift:   06:00 - 14:00
  - Swing shift: 14:00 - 22:00
  - Night shift:  22:00 - 06:00

2. KEY PERFORMANCE INDICATORS

  OEE (Overall Equipment Effectiveness):
    OEE = Availability × Performance × Quality
    Target: ≥ 85%

  Machine Failure Rate:
    = (Units with machine failure) / (Total units produced)
    Target: < 3.5% overall
    Currently: 3.39% in dataset (339/10000)

  Failure Mode Distribution:
    HDF: 33.9% of all failures (most common)
    OSF: 28.9%
    PWF: 28.0%
    TWF: 13.6%
    RNF:  1.5% (irreducible random)

  Mean Time Between Failures (MTBF):
    = Total operating time / Number of failures
    Target: increase quarter over quarter.

  Tool Utilization:
    = Average tool wear at replacement / Max tool life
    Target: 85-95% (replacing too early wastes tools,
    too late risks failure).

3. REPORTING
  - Real-time dashboard: failure rate, sensor readings, alerts.
  - Shift summary: units produced, failures by mode, downtime.
  - Weekly review: trend analysis, MTBF, OEE.
  - Monthly: Pareto analysis of failure modes, corrective actions.

4. CONTINUOUS IMPROVEMENT
  - Track failure rate trend over rolling 30-day window.
  - If rate increases > 0.5pp from baseline → trigger investigation.
  - Prioritize failure modes by frequency × cost impact.
  - HDF and PWF are equipment-related → maintenance owns them.
  - TWF and OSF are process-related → production owns them.
""",
        },
    ]
