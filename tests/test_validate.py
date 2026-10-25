from accusim.validate import main


def test_validation_scorecard_passes(capsys):
    assert main()
    assert "all benchmarks passed" in capsys.readouterr().out
