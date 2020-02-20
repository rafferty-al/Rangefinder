def get_test_runner():
    import unittest
    try:
        from teamcity import is_running_under_teamcity
        from teamcity.unittestpy import TeamcityTestRunner
        from teamcity.unittestpy import TeamcityServiceMessages
        if is_running_under_teamcity():
            runner = TeamcityTestRunner()
        else:
            runner = unittest.TextTestRunner()
    except ModuleNotFoundError:
        runner = unittest.TextTestRunner()
    return runner