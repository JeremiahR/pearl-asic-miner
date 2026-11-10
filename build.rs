//! `cargo build` regenerates every documentation artifact from the model:
//!   docs/figures/*.svg, *.csv     (charts + table views)
//!   docs/figures/*.png, *-dark.png (2x PNG renders, light and dark)
//!   docs/designs.md               (versioned design spec sheets)
//!
//! The model sources are compiled into this build script directly (same
//! modules as the library), so the docs always match the code. Set
//! `PEARL_SKIP_DOCS=1` to skip.

#![allow(dead_code)]

#[path = "src/atom.rs"]
mod atom;
#[path = "src/designs.rs"]
mod designs;
#[path = "src/econ.rs"]
mod econ;
#[path = "src/energy.rs"]
mod energy;
#[path = "src/engine.rs"]
mod engine;
#[path = "src/figures.rs"]
mod figures;
#[path = "src/fp8.rs"]
mod fp8;
#[path = "src/plot.rs"]
mod plot;

use std::path::{Path, PathBuf};

const MODEL_SOURCES: &[&str] = &[
    "build.rs",
    "src/atom.rs",
    "src/designs.rs",
    "src/econ.rs",
    "src/energy.rs",
    "src/engine.rs",
    "src/figures.rs",
    "src/fp8.rs",
    "src/plot.rs",
];

fn main() {
    for f in MODEL_SOURCES {
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-env-changed=PEARL_SKIP_DOCS");
    if std::env::var_os("PEARL_SKIP_DOCS").is_some() {
        return;
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let docs = root.join("docs");
    figures::write_all(&docs).expect("regenerate docs/figures and docs/designs.md");

    let opt = render_options();
    let figs = docs.join("figures");
    let mut svgs: Vec<PathBuf> = std::fs::read_dir(&figs)
        .expect("read docs/figures")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "svg"))
        .collect();
    svgs.sort();
    for svg in svgs {
        let text = std::fs::read_to_string(&svg).unwrap();
        let stem = svg.file_stem().unwrap().to_string_lossy().to_string();
        render_png(&opt, &text, &figs.join(format!("{stem}.png")));
        render_png(&opt, &plot::dark_variant(&text), &figs.join(format!("{stem}-dark.png")));
    }
}

fn render_options() -> resvg::usvg::Options<'static> {
    let mut opt = resvg::usvg::Options::default();
    let db = opt.fontdb_mut();
    db.load_system_fonts();
    // Map the generic sans-serif family to whatever sans font this machine has.
    for family in ["Liberation Sans", "DejaVu Sans", "Noto Sans", "Arial", "Helvetica", "Segoe UI"] {
        let found = db.faces().any(|f| f.families.iter().any(|(n, _)| n == family));
        if found {
            db.set_sans_serif_family(family);
            break;
        }
    }
    opt
}

fn render_png(opt: &resvg::usvg::Options, svg: &str, out: &Path) {
    const SCALE: f32 = 2.0;
    let tree = resvg::usvg::Tree::from_str(svg, opt).expect("parse generated SVG");
    let size = tree.size().to_int_size().scale_by(SCALE).expect("PNG size");
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height()).expect("allocate pixmap");
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(SCALE, SCALE), &mut pixmap.as_mut());
    pixmap.save_png(out).expect("write PNG");
}
