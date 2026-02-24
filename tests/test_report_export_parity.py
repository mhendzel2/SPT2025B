import plotly.graph_objects as go

from enhanced_report_generator import EnhancedSPTReportGenerator


def _build_generator_with_sample_results() -> EnhancedSPTReportGenerator:
    generator = EnhancedSPTReportGenerator()
    generator.report_results = {
        "basic_statistics": {
            "success": True,
            "summary": {"total_tracks": 10, "mean_track_length": 24.5},
            "total_tracks": 10,
            "mean_track_length": 24.5,
        },
        "diffusion_analysis": {
            "success": True,
            "summary": "Synthetic diffusion summary",
            "diffusion_coefficient": 0.123,
        },
    }
    generator.report_figures = {
        "basic_statistics": go.Figure(
            data=[go.Bar(x=["A", "B"], y=[1, 2])],
            layout={"title": "Basic Stats Figure"},
        )
    }
    return generator


def test_collect_report_sections_includes_raw_json_and_figures():
    generator = _build_generator_with_sample_results()
    sections = generator._collect_report_sections(config={"include_raw": True})

    assert len(sections) == 2
    assert sections[0]["title"] == generator.available_analyses["basic_statistics"]["name"]
    assert sections[0]["figures"]
    assert sections[0]["raw_json"] is not None
    assert sections[1]["raw_json"] is not None


def test_html_and_pdf_exports_share_section_coverage():
    generator = _build_generator_with_sample_results()
    units = {"pixel_size": 0.03, "frame_interval": 0.1}
    config = {"include_raw": True}

    sections = generator._collect_report_sections(config=config)
    html_text = generator._export_html_report(config=config, current_units=units).decode("utf-8", errors="replace")
    pdf_bytes = generator._export_pdf_report(current_units=units, config=config)
    assert pdf_bytes is not None
    assert pdf_bytes.startswith(b"%PDF")
    pdf_text = pdf_bytes.decode("latin-1", errors="ignore")

    for section in sections:
        assert section["title"] in html_text
        ascii_title = section["title"].encode("ascii", errors="ignore").decode("ascii").strip()
        if ascii_title:
            assert ascii_title in pdf_text
