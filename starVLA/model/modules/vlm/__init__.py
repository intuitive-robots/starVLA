def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    # Encoder-decoder arm (encdec-vlm). Selected by an explicit flag rather than by
    # base_vlm string, because it is built FROM the same base weights as the causal arm —
    # the name cannot distinguish them.
    if config.framework.qwenvl.get("enc_dec", False):
        # Qwen3.5 needs its own encoder: it is hybrid (linear_attention layers hold a
        # directional DeltaNet, not self_attn), so its encoder mixes a forward and a
        # reversed scan through learned per-layer gates and applies enc_norm/enc_scale.
        # QWen3_EncDec implements none of that and keeps encoder_layers in a different
        # place, so routing a Qwen3.5 checkpoint through it would silently run a different
        # encoder than the one trained. Opt out with qwenvl.encdec_impl.
        impl = str(config.framework.qwenvl.get("encdec_impl", "auto")).lower()
        is_q35 = impl == "qwen35" or (
            impl == "auto" and "qwen3.5" in str(vlm_name).lower()
        )
        if is_q35:
            from .QWen3_5_EncDec import _QWen3_5_EncDec_Interface

            return _QWen3_5_EncDec_Interface(config)

        from .QWen3_EncDec import _QWen3_EncDec_Interface

        return _QWen3_EncDec_Interface(config)

    # Decoder-only arm carrying a mask fine-tuned tail (the causal counterpart of the
    # enc-dec maskft arms). Flag-dispatched for the same reason as enc_dec: base_vlm is
    # identical across arms and cannot distinguish them.
    if config.framework.qwenvl.get("causal_maskft", None):
        from .QWen3_CausalMaskFT import _QWen3_CausalMaskFT_Interface

        return _QWen3_CausalMaskFT_Interface(config)

    # Blind control: same causal backbone, pixel_values zeroed every forward.
    if config.framework.qwenvl.get("blind", False):
        from .QWen3_Blind import _QWen3_Blind_Interface

        return _QWen3_Blind_Interface(config)

    if "Qwen2.5-VL" in vlm_name or "nora" in vlm_name.lower():  # temp for some ckpt
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "Qwen3.5" in vlm_name or "train_downstream" in vlm_name:  # temp for some ckpt
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    elif "gemma-4" in vlm_name.lower() or "gemma4" in vlm_name.lower():
        from .Gemma4 import _Gemma4_VL_Interface

        return _Gemma4_VL_Interface(config)
    elif "molmo2" in vlm_name.lower():
        from .Molmo2 import _Molmo2_VL_Interface

        return _Molmo2_VL_Interface(config)
    elif "florence" in vlm_name.lower():  # temp for some ckpt
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    elif "cosmos-reason2" in vlm_name.lower():
        # Cosmos-Reason2 is architecturally Qwen3-VL (VLM), but implemented
        # in world_model/ for historical reasons. Import directly.
        from starVLA.model.modules.vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
