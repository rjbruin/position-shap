from published_models import GoogleResearchViT, FacebookDeiT, Swin


# NOTE: add implementations here
IMPLEMENTATIONS = {
    'GoogleResearch': GoogleResearchViT,
    'FacebookDeiT': FacebookDeiT,
    'Swin': Swin,
}

def get(model_name, args=None):
    # NOTE: add implementations here
    if model_name == 'GoogleResearch':
        pos_emb_gate_params = {
            'init_value': args.pos_emb_gate_init_value,
            'sigmoid': args.pos_emb_gate_sigmoid,
        }
        return GoogleResearchViT(
            version=args.model_version,
            weights=args.model_weights,
            num_classes=args.num_classes,
            pos_emb=args.pos_emb,
            reset_pe=args.pos_emb_reset,
            pos_emb_gate=args.pos_emb_gate,
            pos_emb_gate_params=pos_emb_gate_params,
        )
    elif model_name == 'FacebookDeiT':
        assert not args.pos_emb_gate, 'FacebookDeiT does not support pos_emb_gate'
        assert not args.pos_emb_reset, 'FacebookDeiT does not support reset_pos_emb'
        return FacebookDeiT(
            version=args.model_version,
            weights=args.model_weights,
            num_classes=args.num_classes,
        )
    elif model_name == 'Swin':
        assert not args.pos_emb_gate, 'Swin does not support pos_emb_gate'
        assert not args.pos_emb_reset, 'Swin does not support reset_pos_emb'
        return Swin(
            version=args.model_version,
            weights=args.model_weights,
            num_classes=args.num_classes,
            pos_emb=args.pos_emb,
        )
    raise NotImplementedError()