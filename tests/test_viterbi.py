import numpy as np
import pytest

import gatk_sv_gd.viterbi as viterbi_module


def test_pair_state_viterbi_rejects_no_candidate_paths(monkeypatch):
    monkeypatch.setattr(viterbi_module, "run_list_viterbi", lambda *args, **kwargs: [])

    with pytest.raises(ValueError, match="sample=S1, cluster=C1"):
        viterbi_module._run_pair_state_viterbi(
            pair_log_priors=np.zeros((1, 1)),
            pair_states=[(1, 1)],
            hap_transition_matrix=np.eye(2),
            breakpoint_mask=None,
            hap_breakpoint_transition_matrix=None,
            sample_id="S1",
            cluster="C1",
        )