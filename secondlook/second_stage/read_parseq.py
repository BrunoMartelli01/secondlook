"""P3 stage 2 (main env, CPU): PARSeq-S on the crops of stage 1 with the qualified adapter (strict checkpoint load). Writes texts and confidences."""
import argparse
import json
from pathlib import Path

import numpy as np

from secondlook.second_stage.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from secondlook.paths import ARTIFACTS

ART = (ARTIFACTS / 'parseq')


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    rec = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=ART / 'parseq-1902db043c029a7e03a3818c616c06600af574be', checkpoint=ART / 'parseq-bb5792a6.pt', device='cpu',
                                                   runtime_source_type=PARSEQ_ARCHIVE_SOURCE, runtime_source_archive=ART / 'parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz'))
    for f in a.files:
        d = np.load(f)
        crops = d['crops']
        out = []
        for i in range(0, len(crops), a.batch):
            for r in rec.recognize_candidate_batch(list(crops[i:i + a.batch])):
                out.append(dict(text=r.raw_text, seq=r.sequence_confidence_native, mint=r.minimum_token_probability, meanlp=r.mean_token_log_probability, n=r.predicted_length))
            if (i // a.batch) % 20 == 0:
                print(Path(f).name, i, len(crops), flush=True)
        Path(str(f).replace('.npz', '_parseq.json')).write_text(json.dumps(dict(owner=d['owner'].tolist(), cidx=d['cidx'].tolist(), rows=out)))
        print('done', Path(f).name, len(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=64)
    main(p.parse_args())
