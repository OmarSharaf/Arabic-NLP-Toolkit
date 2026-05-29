"""
Pytest configuration and shared fixtures.
"""


# Add markers to avoid warnings
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")
