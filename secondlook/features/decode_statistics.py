"""GT-free resolution-benefit signals, including already computed decoder motion."""

import numpy as np

def decode_statistics(language,embeddings,max_len):
    import torch
    n=len(embeddings);pad=language.config.pad_token_id;eos=language.config.eos_token_id
    current=torch.full((n,1),language.config.decoder_start_token_id,dtype=torch.long,device=embeddings.device);finished=torch.zeros(n,dtype=torch.bool,device=embeddings.device)
    count=torch.zeros(n,device=embeddings.device);mass=torch.zeros_like(count);minimum=torch.ones_like(count);entropy=torch.zeros_like(count);margin=torch.ones_like(count);past=None;tokens=[]
    for _ in range(max_len):
        output=language.decoder(input_ids=current,encoder_hidden_states=embeddings,past_key_values=past,use_cache=True);lp=language.lm_head(output.last_hidden_state[:,-1]).log_softmax(-1);p=lp.exp();top,indices=p.topk(2,dim=-1);nxt=indices[:,0]
        active=(~finished)&(nxt!=pad)&(nxt!=eos);count+=active.float();mass+=torch.where(active,top[:,0].log(),torch.zeros_like(count));minimum=torch.where(active,torch.minimum(minimum,top[:,0]),minimum)
        entropy+=torch.where(active,-(p*lp).sum(-1)/np.log(p.shape[-1]),torch.zeros_like(count));margin=torch.where(active,torch.minimum(margin,top[:,0]-top[:,1]),margin)
        nxt=torch.where(finished,torch.full_like(nxt,pad),nxt);tokens.append(nxt);finished=finished|(nxt==pad)|(nxt==eos);past=output.past_key_values;current=nxt[:,None]
        if bool(finished.all()):break
    nonempty=count>0;den=count.clamp_min(1)
    stats=torch.stack([torch.log1p(count),torch.where(nonempty,(mass/den).exp(),torch.zeros_like(count)),torch.where(nonempty,minimum,torch.zeros_like(count)),entropy/den,torch.where(nonempty,margin,torch.zeros_like(count)),finished.float()],dim=1)
    return torch.stack(tokens,1),stats.cpu().numpy()
