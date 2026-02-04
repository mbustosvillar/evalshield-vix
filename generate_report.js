/**
 * generate_report.js
 * ===================
 * Reads the JSON payload from signal_engine.py and produces a professional
 * DOCX tactical report. Designed to be called:
 *   1) Standalone:  node generate_report.js signals.json output.docx
 *   2) Via pipe:    python signal_engine.py | node generate_report.js - output.docx
 *
 * Color palette (professional dark-finance theme):
 *   Background dark:  1A1A2E
 *   Accent gold:      E2B714
 *   Text light:       E8E8E8
 *   Table header:     16213E
 *   Severity colors:  LOW=2E7D32 | MODERATE=F57C00 | HIGH=D32F2F | CRITICAL=B71C1C
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, HeadingLevel
} = require("docx");

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const SEVERITY_COLORS = {
  LOW: { fill: "E8F5E9", text: "2E7D32", recommendations: ["Maintain standard exposure.", "No tactical hedging required."] },
  MODERATE: { fill: "FFF3E0", text: "E65100", recommendations: ["Lighten high-beta/growth exposure.", "Consider covered calls on core positions."] },
  HIGH: { fill: "FFEBEE", text: "C62828", recommendations: ["Tactical hedging via Put options (SPY/QQQ).", "Raise cash levels (10-20%).", "Long volatility overlays."] },
  CRITICAL: { fill: "FCE4EC", text: "B71C1C", recommendations: ["Aggressive de-risking.", "Tail-risk protection (deep OTM puts).", "Preserved liquidity is priority."] }
};

const HEADER_FILL = "16213E";
const HEADER_TEXT = "FFFFFF";
const ALT_ROW_FILL = "F4F6F8";
const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "D0D0D0" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const CELL_MARGINS = { top: 100, bottom: 100, left: 160, right: 160 };

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function headerCell(text, width) {
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: HEADER_FILL, type: ShadingType.CLEAR },
    margins: CELL_MARGINS,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: HEADER_TEXT, size: 22, font: "Arial" })]
    })]
  });
}

function dataCell(text, width, opts = {}) {
  const fill = opts.fill || (opts.altRow ? ALT_ROW_FILL : "FFFFFF");
  const color = opts.color || "333333";
  const bold = opts.bold || false;
  const align = opts.align || AlignmentType.LEFT;
  return new TableCell({
    borders: BORDERS,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: CELL_MARGINS,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, color, size: 21, font: "Arial" })]
    })]
  });
}

function sectionHeading(text) {
  return new Paragraph({
    spacing: { before: 320, after: 160 },
    children: [new TextRun({
      text, bold: true, size: 26, font: "Arial", color: "16213E"
    })]
  });
}

function bodyText(text) {
  return new Paragraph({
    spacing: { after: 120 },
    children: [new TextRun({ text, size: 21, font: "Arial", color: "333333" })]
  });
}

function severityBadge(severity, prob) {
  const s = SEVERITY_COLORS[severity] || SEVERITY_COLORS.LOW;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 80 },
    children: [
      new TextRun({
        text: `▶  AIR POCKET RISK: ${severity}`,
        bold: true, size: 28, font: "Arial", color: s.text
      }),
      new TextRun({
        text: `  |  30D TAIL-RISK PROBABILITY: ${prob}%`,
        bold: true, size: 28, font: "Arial", color: "16213E"
      })
    ]
  });
}

// ─── TABLE BUILDERS ───────────────────────────────────────────────────────────
function buildScoreTable(score) {
  // Columns: Component | Score | Weight | Weighted
  const colWidths = [3500, 1800, 1800, 2160]; // sum = 9260 ≈ letter content width
  const components = [
    { name: "Market Breadth", score: score.breadth_component, weight: 0.25 },
    { name: "VIX / Volatility (Composite)", score: score.vix_component, weight: 0.35 },
    { name: "Momentum", score: score.momentum_component, weight: 0.25 },
    { name: "Yield Curve", score: score.yield_component, weight: 0.15 }
  ];

  const rows = [
    new TableRow({
      children: colWidths.map((w, i) =>
        headerCell(["Component", "Raw Score", "Weight", "Weighted"][i], w)
      )
    })
  ];

  components.forEach((c, i) => {
    const weighted = (c.score * c.weight).toFixed(1);
    const alt = i % 2 === 1;
    rows.push(new TableRow({
      children: [
        dataCell(c.name, colWidths[0], { altRow: alt, bold: true }),
        dataCell(c.score.toFixed(1) + " / 100", colWidths[1], { altRow: alt, align: AlignmentType.CENTER }),
        dataCell((c.weight * 100).toFixed(0) + "%", colWidths[2], { altRow: alt, align: AlignmentType.CENTER }),
        dataCell(weighted, colWidths[3], { altRow: alt, align: AlignmentType.CENTER })
      ]
    }));
  });

  // Total row
  const sev = SEVERITY_COLORS[score.severity] || SEVERITY_COLORS.LOW;
  rows.push(new TableRow({
    children: [
      dataCell("TOTAL SCORE", colWidths[0], { fill: sev.fill, bold: true, color: sev.text }),
      dataCell(score.total.toFixed(1) + " / 100", colWidths[1], { fill: sev.fill, bold: true, color: sev.text, align: AlignmentType.CENTER }),
      dataCell("", colWidths[2], { fill: sev.fill }),
      dataCell(score.severity, colWidths[3], { fill: sev.fill, bold: true, color: sev.text, align: AlignmentType.CENTER })
    ]
  }));

  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: colWidths, rows });
}

function buildTickerTable(signals) {
  const colWidths = [1400, 1600, 1600, 1600, 1400, 1660]; // sum = 9260
  const headers = ["Ticker", "Price", "MA-20", "MA-50", "RSI", "Signal"];

  const rows = [
    new TableRow({ children: colWidths.map((w, i) => headerCell(headers[i], w)) })
  ];

  signals.forEach((s, i) => {
    const alt = i % 2 === 1;
    const sigColor = s.signal_text === "BEARISH" ? "C62828"
      : s.signal_text === "NEUTRAL" ? "F57C00" : "2E7D32";
    const sigFill = s.signal_text === "BEARISH" ? "FFEBEE"
      : s.signal_text === "NEUTRAL" ? "FFF8E1" : "E8F5E9";

    rows.push(new TableRow({
      children: [
        dataCell(s.ticker, colWidths[0], { altRow: alt, bold: true }),
        dataCell("$" + s.price.toFixed(2), colWidths[1], { altRow: alt, align: AlignmentType.RIGHT }),
        dataCell("$" + s.ma_20.toFixed(2), colWidths[2], { altRow: alt, align: AlignmentType.RIGHT }),
        dataCell("$" + s.ma_50.toFixed(2), colWidths[3], { altRow: alt, align: AlignmentType.RIGHT }),
        dataCell(s.rsi.toFixed(1), colWidths[4], { altRow: alt, align: AlignmentType.CENTER }),
        dataCell(s.signal_text, colWidths[5], { fill: sigFill, bold: true, color: sigColor, align: AlignmentType.CENTER })
      ]
    }));
  });

  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: colWidths, rows });
}

function buildContextTable(ctx) {
  const colWidths = [4630, 4630]; // 2-col layout
  const items = [
    ["VIX Current", ctx.vix_current != null ? ctx.vix_current.toFixed(2) : "N/A"],
    ["VIX 3-Month (^VIX3M)", ctx.vix_3m != null ? ctx.vix_3m.toFixed(2) : "N/A"],
    ["VIX Term Structure (V/V3M)", ctx.vix_term_structure != null ? ctx.vix_term_structure.toFixed(3) : "N/A"],
    ["VIX TS Inverted", ctx.vix_term_structure_inverted ? "⚠ YES" : "No"],
    ["VVIX (Vol of Vol)", ctx.vvix_current != null ? ctx.vvix_current.toFixed(2) : "N/A"],
    ["SKEW (Tail Risk Index)", ctx.skew_current != null ? ctx.skew_current.toFixed(2) : "N/A"],
    ["VIX 20-Day MA", ctx.vix_ma20 != null ? ctx.vix_ma20.toFixed(2) : "N/A"],
    ["VIX 50-Day MA", ctx.vix_ma50 != null ? ctx.vix_ma50.toFixed(2) : "N/A"],
    ["VIX 1-Day Change", ctx.vix_1d_change_pct != null ? (ctx.vix_1d_change_pct * 100).toFixed(2) + "%" : "N/A"],
    ["Short-Term (VIX>MA20)", ctx.vix_short_term_signal ? "⚠ YES" : "No"],
    ["Medium-Term (VIX>MA50)", ctx.vix_medium_term_signal ? "⚠ YES" : "No"],
    ["Long-Term (Complacency)", ctx.vix_long_term_signal ? "⚠ YES" : "No"],
    ["TACTICAL SCORE (0-4)", (ctx.tactical_score || 0) + " / 4"],
    ["Breadth (% above MA50)", ((ctx.breadth_pct_above_ma50 || 0) * 100).toFixed(1) + "%"],
    ["Breadth Weak", ctx.breadth_weak ? "⚠ YES" : "No"],
    ["Yield Curve Spread", ctx.yield_curve_spread != null ? ctx.yield_curve_spread.toFixed(3) + "%" : "N/A"],
    ["Yield Curve Inverted", ctx.yield_curve_inverted ? "⚠ YES" : "No"],
    ["Blue Gap (ARS/USD)", ctx.blue_gap_pct != null ? ctx.blue_gap_pct.toFixed(1) + "%" : "N/A"],
    ["Gap Velocity (pts/d)", ctx.gap_velocity != null ? ctx.gap_velocity.toFixed(2) : "0.0"],
    ["DEVAL VACUUM INDEX", ctx.deval_vacuum_index != null ? ctx.deval_vacuum_index.toFixed(1) : "N/A"],
    ["X Sentiment Score", ctx.x_sentiment_score != null ? ctx.x_sentiment_score.toFixed(2) : "N/A"],
    ["X Sentiment Label", ctx.x_sentiment_label || "N/A"]
  ];

  const rows = [
    new TableRow({
      children: [
        headerCell("Indicator", colWidths[0]),
        headerCell("Reading", colWidths[1])
      ]
    })
  ];

  items.forEach((item, i) => {
    const alt = i % 2 === 1;
    const isWarning = String(item[1]).includes("⚠");
    rows.push(new TableRow({
      children: [
        dataCell(item[0], colWidths[0], { altRow: alt, bold: true }),
        dataCell(item[1], colWidths[1], { altRow: alt, color: isWarning ? "C62828" : "333333", bold: isWarning })
      ]
    }));
  });

  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: colWidths, rows });
}

function buildRegimeTable(probs) {
  const colWidths = [4630, 4630];
  const rows = [
    new TableRow({
      children: [
        headerCell("Market Regime", colWidths[0]),
        headerCell("Probability", colWidths[1])
      ]
    }),
    new TableRow({
      children: [
        dataCell("EXPANSION (Low Risk)", colWidths[0], { bold: true }),
        dataCell((probs.expansion * 100).toFixed(1) + "%", colWidths[1], { align: AlignmentType.CENTER })
      ]
    }),
    new TableRow({
      children: [
        dataCell("FRAGILE (Correction Risk)", colWidths[0], { bold: true, color: "E65100" }),
        dataCell((probs.fragile * 100).toFixed(1) + "%", colWidths[1], { align: AlignmentType.CENTER, color: "E65100" })
      ]
    }),
    new TableRow({
      children: [
        dataCell("STRESS (Systemic Risk)", colWidths[0], { bold: true, color: "C62828" }),
        dataCell((probs.stress * 100).toFixed(1) + "%", colWidths[1], { align: AlignmentType.CENTER, color: "C62828" })
      ]
    }),
    new TableRow({
      children: [
        dataCell("TAIL RISK (Black Swan)", colWidths[0], { bold: true, color: "B71C1C", fill: "FCE4EC" }),
        dataCell((probs.tail_risk * 100).toFixed(1) + "%", colWidths[1], { align: AlignmentType.CENTER, color: "B71C1C", fill: "FCE4EC", bold: true })
      ]
    })
  ];
  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: colWidths, rows });
}

function buildAllocationTable(allocation) {
  const colWidths = [4630, 4630];
  const rows = [
    new TableRow({
      children: [
        headerCell("Asset Class / Product", colWidths[0]),
        headerCell("Target Weight (%)", colWidths[1])
      ]
    })
  ];

  Object.entries(allocation).forEach(([asset, weight], i) => {
    const alt = i % 2 === 1;
    rows.push(new TableRow({
      children: [
        dataCell(asset, colWidths[0], { altRow: alt, bold: true }),
        dataCell(weight + "%", colWidths[1], { altRow: alt, align: AlignmentType.CENTER, bold: true, color: "16213E" })
      ]
    }));
  });

  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: colWidths, rows });
}

// ─── INTERPRETATION ENGINE ────────────────────────────────────────────────────
function generateInterpretation(score, ctx) {
  const lines = [];
  const sev = score.severity;

  // Lead with severity framing
  if (ctx.tactical_score >= 3) {
    lines.push("CRITICAL CONFLUENCE: Short, Medium, and Long-term signals are all active. Historical analogs suggest extreme caution. The VIX is trending above its medium-term averages while starting from a complacency floor.");
  } else if (ctx.tactical_score === 2) {
    lines.push("ELEVATED RISK: Two out of three VIX tactical signals are active. Internal market structure is weakening relative to recent volatility benchmarks.");
  } else if (ctx.tactical_score === 1) {
    lines.push("CAUTION: A single tactical regime shift is detected. Monitor for secondary confirmation.");
  } else {
    lines.push("STABLE REGIME: No tactical VIX signals are currently active. Volatility dynamics remain within standard ranges.");
  }

  // Flag-specific commentary
  const flags = score.condition_flags;
  if (flags.includes("BREADTH_DETERIORATION")) {
    lines.push("Breadth is deteriorating: fewer than " + (45) + "% of tracked ETFs are above their 50-day MA. This is a classic leading indicator — price indices can remain elevated while internal weakness accumulates.");
  }
  if (flags.includes("VIX_COMPLACENCY")) {
    lines.push("VIX is in complacency territory. Sustained low-volatility regimes compress risk premiums and encourage leverage — the exact conditions that amplify the magnitude of subsequent corrections.");
  }
  if (flags.includes("VIX_PANIC_SPIKE")) {
    lines.push("VIX recorded a single-day spike exceeding the panic threshold. This typically signals institutional de-risking and forced liquidation, not orderly selling.");
  }
  if (flags.includes("VIX_RISING")) {
    lines.push("VIX is rising but has not yet breached the panic threshold. Monitor for continuation.");
  }
  if (flags.includes("YIELD_CURVE_INVERTED")) {
    lines.push("The yield curve is inverted (10Y < 2Y). Historically, sustained inversion has preceded every major recession in the past 50 years. Lead time is variable (6–18 months), but the signal is active.");
  }
  if (flags.includes("VIX_TERM_STRUCTURE_INVERTED")) {
    lines.push("VIX Term Structure is INVERTED (VIX > VIX3M). This is an institutional-grade danger signal, often associated with forced liquidation and extreme tail-risk events.");
  }
  if (flags.includes("VVIX_EXCESSIVE")) {
    lines.push("VVIX is exceptionally high (>110). Panic is spreading through the options-pricing engine, suggesting that the 'volatility of volatility' is spiking.");
  }
  if (flags.includes("VVIX_ELEVATED")) {
    lines.push("VVIX is elevated (>95). Increased uncertainty in volatility expectations usually precedes larger price swings.");
  }
  if (flags.includes("MAJORITY_BEARISH")) {
    lines.push("The majority of tracked instruments are registering bearish momentum. Broad-based weakness increases the probability of correlated selloffs.");
  }

  return lines;
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);
  const inFile = args[0] || "signals.json";
  const outFile = args[1] || "tactical_report.docx";

  // Read input: stdin (-) or file
  let raw;
  if (inFile === "-") {
    raw = fs.readFileSync(0, "utf8");  // read stdin
  } else {
    raw = fs.readFileSync(inFile, "utf8");
  }
  const data = JSON.parse(raw);

  // Support for merged macro+signals or legacy format
  const { context: ctx, score, ticker_signals: signals, generated_at, macro_report } = data;
  const genDate = new Date(generated_at);
  const dateStr = genDate.toUTCString();

  const interpretation = generateInterpretation(score, ctx);

  // ─── DOCUMENT ASSEMBLY ──────────────────────────────────────────────────────
  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Arial", size: 22 } } }
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [new TextRun({ text: "TACTICAL MARKET REPORT  |  AIR POCKET ANALYSIS", size: 18, color: "999999", font: "Arial" })]
            })
          ]
        })
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Generated: " + dateStr + "  |  Page ", size: 18, color: "999999", font: "Arial" }),
                new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "999999", font: "Arial" })
              ]
            })
          ]
        })
      },
      children: [
        // ── TITLE BLOCK ─────────────────────────────────────────────────────
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 40 },
          children: [new TextRun({ text: "TACTICAL MARKET REPORT", bold: true, size: 36, font: "Arial", color: "16213E" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
          children: [new TextRun({ text: "Air Pocket Detection & Signal Analysis", size: 24, font: "Arial", color: "666666" })]
        }),

        // ── SEVERITY BADGE ──────────────────────────────────────────────────
        severityBadge(score.severity, data.tail_risk_probability || "N/A"),

        new Paragraph({ spacing: { after: 200 }, children: [] }),

        // ── SECTION 0: MACRO REGIME (LAYER 1) ───────────────────────────────
        ...(macro_report ? [
          sectionHeading("0. Probabilistic Macro Regime Classification"),
          bodyText(`Current regime detected: ${macro_report.regime}. This classification is based on real-time fusion of VIX term structure, cross-asset volatility (MOVE/SKEW), and market breadth internals.`),
          buildRegimeTable(macro_report.probabilities),
          new Paragraph({ spacing: { after: 200 }, children: [] })
        ] : []),

        // ── SECTION 1: COMPOSITE SCORE ──────────────────────────────────────
        sectionHeading("1. Composite Air Pocket Score"),
        bodyText(`The composite score aggregates four independent signal dimensions into a single 0–100 risk metric. Scores above 50 indicate elevated air pocket probability based on current market internals.`),
        buildScoreTable(score),

        new Paragraph({ spacing: { after: 120 }, children: [] }),

        // ── SECTION 2: ACTIVE FLAGS ─────────────────────────────────────────
        sectionHeading("2. Active Condition Flags"),
        ...(score.condition_flags.length > 0
          ? score.condition_flags.map(f =>
            new Paragraph({
              spacing: { after: 60 },
              children: [
                new TextRun({ text: "▸ ", bold: true, size: 22, font: "Arial", color: "C62828" }),
                new TextRun({ text: f, bold: true, size: 22, font: "Arial", color: "333333" })
              ]
            })
          )
          : [bodyText("No condition flags currently active.")]
        ),

        new Paragraph({ spacing: { after: 120 }, children: [] }),

        // ── SECTION 3: MARKET CONTEXT ───────────────────────────────────────
        sectionHeading("3. Market Context"),
        bodyText("Real-time readings across volatility, breadth, and macro indicators."),
        buildContextTable(ctx),

        new Paragraph({ children: [new PageBreak()] }),

        // ── SECTION 4: TICKER-LEVEL SIGNALS ─────────────────────────────────
        sectionHeading("4. Ticker-Level Signals"),
        bodyText(`Individual instrument analysis: price relative to 20-day and 50-day moving averages, RSI momentum, and directional classification.`),
        buildTickerTable(signals),

        new Paragraph({ spacing: { after: 200 }, children: [] }),

        // ── SECTION 5: DEVALSHIELD NARRATIVE ────────────────────────────────
        sectionHeading("5. DevalShield AR: Strategic Narrative"),
        ...(data.narrative && data.narrative.length > 0
          ? data.narrative.map(line => bodyText(line))
          : [bodyText("No strategic narrative generated for the current regime.")]),

        new Paragraph({ spacing: { after: 200 }, children: [] }),

        // ── SECTION 6: TACTICAL ALLOCATION ───────────────────────────────
        sectionHeading("6. Layer 3: RL-Optimized Tactical Allocation"),
        bodyText("Recommended capital deployment based on the Reinforcement Learning agent's policy, optimizing for Sortino-asymmetry and drawdown protection."),
        buildAllocationTable(data.rl_allocation || {}),

        new Paragraph({ spacing: { after: 280 }, children: [] }),

        // ── SECTION 7: HEDGING RECOMMENDATIONS ──────────────────────────────
        sectionHeading("7. Tactical Hedging & CFA Recommendations"),
        bodyText(`Based on the current Air Pocket Score of ${score.total.toFixed(1)}, the following tactical adjustments are advised:`),
        ...(SEVERITY_COLORS[score.severity] || SEVERITY_COLORS.LOW).recommendations.map(r =>
          new Paragraph({
            spacing: { after: 60 },
            children: [
              new TextRun({ text: "▸ ", bold: true, size: 22, font: "Arial", color: (SEVERITY_COLORS[score.severity] || SEVERITY_COLORS.LOW).text }),
              new TextRun({ text: r, size: 22, font: "Arial", color: "333333" })
            ]
          })
        ),

        new Paragraph({ spacing: { after: 200 }, children: [] }),

        // ── SECTION 7: METHODOLOGY ──────────────────────────────────────────
        sectionHeading("7. Methodology & Layer 2 Vision"),
        bodyText("This report is generated by signal_engine.py (Layer 2 - Early Warning Prototype)."),
        bodyText("New Capacities: VIX Term Structure, VVIX, SKEW. Scoring weights: Breadth 25%, VIX (Composite) 35%, Momentum 25%, Yield Curve 15%."),
        bodyText("Limitations: (1) Breadth is proxied via ETF watchlist, not full A/D line. (2) Yield curve 2Y data may be unavailable via yfinance — check context table for gaps. (3) This is a signal aggregator, not a trading system. No position sizing, execution, or risk-management logic is included. (4) All data is subject to feed delays and corporate actions."),
        bodyText("This report is for informational purposes only and does not constitute investment advice.")
      ]
    }]
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(outFile, buffer);
  console.error(`[INFO] Report written to ${outFile}`);
}

main().catch(e => { console.error(e); process.exit(1); });
