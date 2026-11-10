//! Minimal dependency-free SVG chart primitives.
//!
//! Charts are static SVG for GitHub markdown. Each file carries its own
//! light/dark tokens (validated categorical slots 1-4 of the reference palette)
//! via `prefers-color-scheme`. Every mark is direct-labelled and each figure
//! ships a CSV table view beside it.

use std::fmt::Write;

pub const FONT: &str = "system-ui, -apple-system, 'Segoe UI', sans-serif";

struct Tokens {
    surface: &'static str,
    ink1: &'static str,
    ink2: &'static str,
    muted: &'static str,
    grid: &'static str,
    base: &'static str,
    series: [&'static str; 4],
    ink_strong: &'static str,
    wash: &'static str,
}

const LIGHT: Tokens = Tokens {
    surface: "#fcfcfb",
    ink1: "#0b0b0b",
    ink2: "#52514e",
    muted: "#898781",
    grid: "#e1e0d9",
    base: "#c3c2b7",
    series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    ink_strong: "#0b0b0b",
    wash: "rgba(11,11,11,0.05)",
};

const DARK: Tokens = Tokens {
    surface: "#1a1a19",
    ink1: "#f0efec",
    ink2: "#c3c2b7",
    muted: "#898781",
    grid: "#2c2c2a",
    base: "#383835",
    series: ["#3987e5", "#d95926", "#199e70", "#c98500"],
    ink_strong: "#f0efec",
    wash: "rgba(255,255,255,0.06)",
};

/// Colour rules for one theme. Plain per-class colours (no CSS custom
/// properties) so every SVG renderer, including librsvg, draws them.
fn color_rules(t: &Tokens) -> String {
    let [s1, s2, s3, s4] = t.series;
    format!(
        ".bg{{fill:{sf}}} .title,.lab,.val{{fill:{i1}}} .sub,.lab2{{fill:{i2}}} .tick,.note{{fill:{mu}}} \
         .grid{{stroke:{gr}}} .base{{stroke:{ba}}} .ref{{stroke:{i2}}} .wash{{fill:{wa}}} .ring{{stroke:{sf}}} \
         .f1{{fill:{s1}}} .f2{{fill:{s2}}} .f3{{fill:{s3}}} .f4{{fill:{s4}}} \
         .l1,.w1{{stroke:{s1}}} .l2,.w2{{stroke:{s2}}} .l3{{stroke:{s3}}} \
         .b1{{fill:{s1};fill-opacity:0.2}} .b2{{fill:{s2};fill-opacity:0.2}} .b3{{fill:{s3};fill-opacity:0.2}} \
         .refbold{{stroke:{is}}} .strong,.region{{fill:{is}}} .regionsub{{fill:{i2}}}",
        is = t.ink_strong,
        sf = t.surface, i1 = t.ink1, i2 = t.ink2, mu = t.muted, gr = t.grid, ba = t.base, wa = t.wash
    )
}

fn style() -> String {
    format!(
        "text{{font-family:{FONT}}} .title{{font-size:16px;font-weight:600}} .sub{{font-size:12px}} \
         .lab{{font-size:12px}} .lab2{{font-size:11px}} .val{{font-size:11.5px;font-variant-numeric:tabular-nums}} \
         .tick{{font-size:11px;font-variant-numeric:tabular-nums}} .note{{font-size:11px}} \
         .grid,.base,.ref{{stroke-width:1}} .ring{{stroke-width:2}} \
         .l1,.l2,.l3{{stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}} \
         .refbold{{stroke-width:2.5}} .strong{{font-size:12.5px;font-weight:600}} .region{{font-size:22px;font-weight:700;letter-spacing:0.5px}} .regionsub{{font-size:12px}} .b1,.b2,.b3{{stroke:none}} \
         .w1,.w2{{stroke-width:2;stroke-linecap:round}}\n{}\n@media (prefers-color-scheme: dark) {{ {} }}",
        color_rules(&LIGHT),
        color_rules(&DARK)
    )
}

pub struct Svg {
    pub w: f64,
    pub h: f64,
    body: String,
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

impl Svg {
    pub fn new(w: f64, h: f64, title: &str, subtitle: &str) -> Self {
        let mut s = Svg { w, h, body: String::new() };
        s.rect(0.0, 0.0, w, h, 0.0, "bg");
        s.text(20.0, 28.0, title, "title", "start");
        s.text(20.0, 47.0, subtitle, "sub", "start");
        s
    }
    pub fn text(&mut self, x: f64, y: f64, t: &str, class: &str, anchor: &str) {
        let _ = writeln!(self.body, r#"<text x="{x:.1}" y="{y:.1}" class="{class}" text-anchor="{anchor}">{}</text>"#, esc(t));
    }
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, class: &str) {
        let _ = writeln!(self.body, r#"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" class="{class}"/>"#);
    }
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, rx: f64, class: &str) {
        let _ = writeln!(self.body, r#"<rect x="{x:.1}" y="{y:.1}" width="{:.1}" height="{h:.1}" rx="{rx}" class="{class}"/>"#, w.max(0.0));
    }
    /// Bar segment with only the right-hand (data) end rounded.
    pub fn bar_end_rounded(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64, class: &str) {
        let r = r.min(w / 2.0).min(h / 2.0);
        let (x2, y2) = (x + w, y + h);
        let _ = writeln!(
            self.body,
            r#"<path d="M{x:.1},{y:.1} H{:.1} Q{x2:.1},{y:.1} {x2:.1},{:.1} V{:.1} Q{x2:.1},{y2:.1} {:.1},{y2:.1} H{x:.1} Z" class="{class}"/>"#,
            x2 - r,
            y + r,
            y2 - r,
            x2 - r
        );
    }
    pub fn dot(&mut self, cx: f64, cy: f64, r: f64, fill_class: &str) {
        let _ = writeln!(self.body, r#"<circle cx="{cx:.1}" cy="{cy:.1}" r="{r}" class="{fill_class} ring"/>"#);
    }
    pub fn polyline(&mut self, pts: &[(f64, f64)], class: &str) {
        let p: Vec<String> = pts.iter().map(|(x, y)| format!("{x:.1},{y:.1}")).collect();
        let _ = writeln!(self.body, r#"<polyline points="{}" class="{class}"/>"#, p.join(" "));
    }
    pub fn polygon(&mut self, pts: &[(f64, f64)], class: &str) {
        let p: Vec<String> = pts.iter().map(|(x, y)| format!("{x:.1},{y:.1}")).collect();
        let _ = writeln!(self.body, r#"<polygon points="{}" class="{class}"/>"#, p.join(" "));
    }
    /// Legend row of (fill class, label) swatches starting at (x, y).
    pub fn legend(&mut self, x: f64, y: f64, items: &[(&str, &str)]) {
        let mut cx = x;
        for (class, label) in items {
            self.rect(cx, y - 9.0, 10.0, 10.0, 2.0, class);
            self.text(cx + 15.0, y, label, "lab", "start");
            cx += 22.0 + 6.6 * label.len() as f64;
        }
    }
    pub fn finish(self, desc: &str) -> String {
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" role=\"img\" aria-label=\"{d}\">\n<title>{d}</title>\n<style>{st}</style>\n{b}</svg>\n",
            w = self.w,
            h = self.h,
            d = esc(desc),
            st = style(),
            b = self.body
        )
    }
}

const DARK_MEDIA: &str = "@media (prefers-color-scheme: dark) { ";

/// The same SVG with the dark palette applied unconditionally (for PNG renders,
/// which have no `prefers-color-scheme`).
pub fn dark_variant(svg: &str) -> String {
    match svg.find(DARK_MEDIA) {
        Some(i) => {
            let rest = &svg[i + DARK_MEDIA.len()..];
            let end = rest.find(" }</style>").expect("dark media block closes before </style>");
            format!("{}{}{}", &svg[..i], &rest[..end], &rest[end + 2..])
        }
        None => svg.to_string(),
    }
}

#[derive(Clone, Copy)]
pub struct LogAxis {
    pub lo: f64,
    pub hi: f64,
    pub a: f64,
    pub b: f64,
}
impl LogAxis {
    pub fn map(&self, v: f64) -> f64 {
        let t = (v.log10() - self.lo.log10()) / (self.hi.log10() - self.lo.log10());
        self.a + t.clamp(0.0, 1.0) * (self.b - self.a)
    }
}

#[derive(Clone, Copy)]
pub struct LinAxis {
    pub lo: f64,
    pub hi: f64,
    pub a: f64,
    pub b: f64,
}
impl LinAxis {
    pub fn map(&self, v: f64) -> f64 {
        self.a + (v - self.lo) / (self.hi - self.lo) * (self.b - self.a)
    }
}

/// Axis tick for powers/multiples of ten without trailing zeros ("0.0001", "0.5", "10").
pub fn fmt_tick(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

/// Compact number formatting for labels ("0.00042", "1,979", "63").
pub fn fmt_num(v: f64) -> String {
    if v >= 1000.0 {
        let s = format!("{:.0}", v);
        let mut out = String::new();
        for (i, c) in s.chars().enumerate() {
            if i > 0 && (s.len() - i).is_multiple_of(3) {
                out.push(',');
            }
            out.push(c);
        }
        out
    } else if v >= 10.0 {
        format!("{v:.0}")
    } else if v >= 1.0 {
        format!("{v:.1}")
    } else {
        let digits = (-v.log10()).ceil() as usize + 1;
        format!("{v:.digits$}")
    }
}
