from pathlib import Path
import sys
import types


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


def _install_test_stub_modules():
    if "tqdm" not in sys.modules:
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda iterable=None, *args, **kwargs: iterable
        sys.modules["tqdm"] = tqdm

    if "intervaltree" not in sys.modules:
        intervaltree = types.ModuleType("intervaltree")

        class _StubIntervalTree:
            def __init__(self, *args, **kwargs):
                pass

            def addi(self, *args, **kwargs):
                return None

            def merge_overlaps(self, *args, **kwargs):
                return None

            def overlap(self, *args, **kwargs):
                return []

            def __len__(self):
                return 0

        intervaltree.IntervalTree = _StubIntervalTree
        sys.modules["intervaltree"] = intervaltree

    if "pysam" not in sys.modules:
        pysam = types.ModuleType("pysam")

        class _StubTabixFile:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("TabixFile stub must be monkeypatched in tests")

        class _StubVariantRecord:
            pass

        class _StubVariantHeader:
            pass

        class _StubVariantFile:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("VariantFile stub must be monkeypatched in tests")

        pysam.TabixFile = _StubTabixFile
        pysam.VariantRecord = _StubVariantRecord
        pysam.VariantHeader = _StubVariantHeader
        pysam.VariantFile = _StubVariantFile
        pysam.tabix_compress = lambda *args, **kwargs: None
        pysam.tabix_index = lambda *args, **kwargs: None
        sys.modules["pysam"] = pysam

    if "torch" not in sys.modules:
        try:
            __import__("torch")
        except ImportError:
            torch = types.ModuleType("torch")

            class _StubTensor:
                pass

            class _StubDType:
                pass

            torch.Tensor = _StubTensor
            torch.dtype = _StubDType
            torch.float32 = _StubDType()
            torch.int32 = _StubDType()
            torch.optim = types.SimpleNamespace(Adam=object)
            sys.modules["torch"] = torch

    if "pyro" in sys.modules:
        return

    pyro = types.ModuleType("pyro")
    pyro.distributions = types.ModuleType("pyro.distributions")
    pyro.poutine = types.ModuleType("pyro.poutine")
    pyro.clear_param_store = lambda: None
    pyro.enable_validation = lambda value: None
    pyro.distributions.enable_validation = lambda value: None
    pyro.set_rng_seed = lambda value: None

    pyro_ops = types.ModuleType("pyro.ops")
    pyro_ops_indexing = types.ModuleType("pyro.ops.indexing")
    pyro_ops_indexing.Vindex = object()
    pyro.ops = pyro_ops

    pyro_optim = types.ModuleType("pyro.optim")
    pyro_optim.LambdaLR = object
    pyro.optim = pyro_optim

    pyro_infer = types.ModuleType("pyro.infer")

    def _config_enumerate(fn=None, *args, **kwargs):
        if fn is None:
            return lambda inner_fn: inner_fn
        return fn

    pyro_infer.config_enumerate = _config_enumerate
    pyro_infer.infer_discrete = lambda *args, **kwargs: None
    pyro_infer.JitTraceEnum_ELBO = object
    pyro_infer.TraceEnum_ELBO = object

    pyro_autoguide = types.ModuleType("pyro.infer.autoguide")

    class _StubAutoGuideList:
        def __init__(self, *args, **kwargs):
            self.guides = []

        def append(self, guide):
            self.guides.append(guide)

    pyro_autoguide.AutoDiagonalNormal = object
    pyro_autoguide.AutoDelta = object
    pyro_autoguide.AutoGuideList = _StubAutoGuideList

    pyro_autoguide_initialization = types.ModuleType(
        "pyro.infer.autoguide.initialization"
    )
    pyro_autoguide_initialization.init_to_value = lambda *args, **kwargs: None

    pyro_svi = types.ModuleType("pyro.infer.svi")
    pyro_svi.SVI = object

    sys.modules["pyro"] = pyro
    sys.modules["pyro.distributions"] = pyro.distributions
    sys.modules["pyro.poutine"] = pyro.poutine
    sys.modules["pyro.ops"] = pyro_ops
    sys.modules["pyro.ops.indexing"] = pyro_ops_indexing
    sys.modules["pyro.optim"] = pyro_optim
    sys.modules["pyro.infer"] = pyro_infer
    sys.modules["pyro.infer.autoguide"] = pyro_autoguide
    sys.modules["pyro.infer.autoguide.initialization"] = pyro_autoguide_initialization
    sys.modules["pyro.infer.svi"] = pyro_svi


_install_test_stub_modules()
