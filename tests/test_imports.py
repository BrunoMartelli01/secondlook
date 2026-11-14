"""Every module of the package imports (needs the analysis environment of requirements.txt, no data)."""
import importlib
import pkgutil
import unittest

# scripts that read sys.argv or create the run view directory when they are imported
SCRIPT_STYLE = {'secondlook.figures.make_fig56', 'secondlook.second_stage.read_native', 'secondlook.reruns.exact.report'}


class ImportAll(unittest.TestCase):
    def test_every_module_imports(self):
        import secondlook
        failed = []
        for m in pkgutil.walk_packages(secondlook.__path__, 'secondlook.'):
            if m.name in SCRIPT_STYLE:
                continue
            try:
                importlib.import_module(m.name)
            except Exception as e:   # collect every failure, then report them together
                failed.append((m.name, repr(e)))
        self.assertEqual(failed, [])


if __name__ == '__main__':
    unittest.main()
