def test_single_package_entrypoint():
    from importlib.util import find_spec
    assert find_spec('shellshock') is not None
    from shellshock.config.paths import DATA_ROOT
    assert (DATA_ROOT / 'runs').exists()
