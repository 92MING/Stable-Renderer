
def calc_cond_uncond_batch(model: model_base.BaseModel, 
                           cond: list["ConvertedCondition"], 
                           uncond: list["ConvertedCondition"], 
                           x_in: Tensor, 
                           timestep: Tensor,    # e.g. Tensor([0.7297]), the ratio of the current timestep to the total timesteps
                           model_options: dict,     # e.g. {'transformer_options': ...}
                           **kwargs):
    '''
    This method concat the cond & uncond's inference input as a batch, i.e. (2*4*64*64),
    and return both's result(latent space)
    '''
    
    # here, cond[0]['model_conds']['c_crossattn'].cond.shape is still [1, 77, 768]
    if is_dev_mode() and is_verbose_mode():
        pos_cross_attn_shapes = [c['model_conds']['c_crossattn'].cond.shape for c in cond]
        neg_cross_attn_shapes = [c['model_conds']['c_crossattn'].cond.shape for c in uncond]
        ComfyUILogger.debug(f'(comfy.samplers.calc_cond_uncond_batch) pos_cross_attn_shapes={pos_cross_attn_shapes}')
        ComfyUILogger.debug(f'(comfy.samplers.calc_cond_uncond_batch) neg_cross_attn_shapes={neg_cross_attn_shapes}')
    if cond is None:
        cond = []
    if uncond is None:
        uncond = []
    
    batch_size = x_in.shape[0]
    input_latent_shape = [x_in.shape[-2], x_in.shape[-1]]  # height, width
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1
    to_run: list[tuple[ConditionObj, Literal[0, 1]]] = []
    
    individual_conds, individual_unconds = [], []
    real_conds, real_unconds = [], []
    for i in tuple(range(len(cond))):
        c = cond[i]
        if c.get('individual_cond', False):
            individual_conds.append(c)
        else:
            real_conds.append(c)
    for i in  tuple(range(len(uncond))):
        c = uncond[i]
        if c.get('individual_cond', False):
            individual_unconds.append(c)
        else:
            real_unconds.append(c)
    
    if is_dev_mode() and is_verbose_mode():
        ComfyUILogger.debug(f'(comfy.samplers.calc_cond_uncond_batch) individual_conds={len(individual_conds)}, cond={len(real_conds)}')
        ComfyUILogger.debug(f'(comfy.samplers.calc_cond_uncond_batch) individual_unconds={len(individual_unconds)}, uncond={len(real_unconds)}')
        
    if len(individual_conds)>0 and len(individual_conds) < x_in.shape[0]:
        individual_conds += [individual_conds[-1]] * (x_in.shape[0] - len(individual_conds))    # repeat the last one to batch_size
    for i, individual_cond in enumerate(individual_conds):
        p = get_area_and_mult(individual_cond, x_in[i].clone(), timestep, repeat_to_batch=False, condition_type='pos')
        if p is None:
            continue
        to_run += [(p, COND)]
        
    if len(individual_unconds)>0 and len(individual_unconds) < x_in.shape[0]:
        individual_unconds += [individual_unconds[-1]] * (x_in.shape[0] - len(individual_unconds))    # repeat the last one to batch_size
    for i, individual_uncond in enumerate(individual_unconds):
        p = get_area_and_mult(individual_uncond, x_in[i].clone(), timestep, repeat_to_batch=False, condition_type='neg')
        if p is None:
            continue
        to_run += [(p, UNCOND)]
    
    for i in real_conds:
        p = get_area_and_mult(i, x_in, timestep, condition_type='pos')
        # in normal case, `get_area_and_mult` will make the origin condition tensor repeated to batch_size
        if p is None:
            continue
        to_run += [(p, COND)]
        
    for i in real_unconds:
        p = get_area_and_mult(i, x_in, timestep, condition_type='neg')
        if p is None:
            continue
        to_run += [(p, UNCOND)]
    print('to run len:', len(to_run))
    while len(to_run) > 0:
        first_cond = to_run[0]
        first_latent_shape = first_cond[0].input_x.shape
        
        to_batch_indices: list[int] = []   # can contains both pos or neg jobs
        
        for i in range(len(to_run)):
            if can_concat_cond(to_run[i][0], first_cond[0]):
                to_batch_indices += [i]

        to_batch_indices.reverse()
        to_batch = to_batch_indices[:1] # start from last one

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_indices) + 1):   # from 1 to len(to_batch_indices), test the max possible batch size can be run
            possible_batch_indices = to_batch_indices[:len(to_batch_indices)//i]
            input_shape = [len(possible_batch_indices) * first_latent_shape[0]] + list(first_latent_shape)[1:]    # e.g. [2,] + [4, 64, 64] = [2, 4, 64, 64]
            if model.memory_required(input_shape) < free_memory:    # type: ignore
                to_batch = possible_batch_indices
                break

        all_input_x = []    # all latents
        mult = []
        conditions: list[dict[str, "CONDRegular"]] = []
        cond_or_uncond: list[Literal[0, 1]] = []    # e.g. [COND, UNCOND, COND, COND, UNCOND, ...]
        positive_cond_indices: list[int] = []
        area: list[tuple[int,int,int,int]] = []
        control: Optional["ControlBase"] = None
        patches: Optional[dict[str, Any]] = None    # the data for modifying the model's patches
        
        print(to_batch)
        for i in to_batch:
            cond_obj, condition_type = to_run.pop(i)
            
            if condition_type == COND:
                positive_cond_indices.extend(list(range(len(all_input_x), len(all_input_x) + cond_obj.input_x.shape[0])))
            
            all_input_x.append(cond_obj.input_x)
            mult.append(cond_obj.mult)
            conditions.append(cond_obj.conditioning)
            area.append(cond_obj.area)
            control = cond_obj.control
            patches = cond_obj.patches
            cond_or_uncond.append(condition_type)
        
        batch_chunks = len(cond_or_uncond)	 
        batch_chunk_indices: list[list[int]] = []
        offset = 0
        for inp in all_input_x:
            batch_chunk_indices.append(list(range(offset, offset+inp.shape[0])))
            offset += inp.shape[0]
        
        input_x = torch.cat(all_input_x)
        c = cond_cat(conditions)    # e.g. for 1 pos, 2 neg, batch_size=3, here c['c_crossattn'].shape = [9, 77, 768], 9=3*(1+2)
        timestep_ = torch.cat([timestep] * len(cond_or_uncond))[:len(input_x)] # for case existing individual conditions, timestep should be duplicated, so need to slice it
        
        print(f'batch_chunks={batch_chunks}')
        print(f'input_x.shape={input_x.shape}')
        print(f'c.shape={c["c_crossattn"].shape}')
        print(f'timestep_.shape={timestep_.shape}')
        
        if control is not None:
            c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))
            
        if is_dev_mode() and is_verbose_mode():
            prompt_embed_shape = c['c_crossattn'].shape
            ComfyUILogger.debug(f'(calc_cond_uncond_batch) input_x.shape={input_x.shape}, prompt_embed_shape={prompt_embed_shape}, timestep_shape={timestep_.shape}')
            ComfyUILogger.debug(f'(calc_cond_uncond_batch) cond_or_uncond={cond_or_uncond}, positive_cond_indices={positive_cond_indices}, area={area}')
            ComfyUILogger.debug(f'(calc_cond_uncond_batch) batch_chunks={len(batch_chunk_indices)}, batch_indices={batch_chunk_indices}')
        
        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options['positive_cond_indices'] = positive_cond_indices
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options
        c.update(kwargs)    # extra args are passed to the model from here.

        if 'model_function_wrapper' in model_options:
            all_output: Tensor = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond})
        else:
            all_output: Tensor = model.apply_model(input_x, timestep_, **c)     # BaseModel.apply_model here
        #output = all_output.chunk(batch_chunks)
        output: list[Tensor] = []
        for batch_chunk_idx in batch_chunk_indices:
            output.append(all_output[batch_chunk_idx].contiguous().clone())
        print(len(output), [o.shape for o in output])
        del input_x
        
        #for o in range(batch_chunks):
        tmp = None
        tmp_mult = None
        for o in range(len(output)):
            print('cond_or_uncond:', cond_or_uncond[o], 'output:', output[o].shape, 'mult:', mult[o].shape, 'area:', area[o])
            if tmp is not None:
                tmp = torch.cat([tmp, output[o]], 0)
                tmp_mult = torch.cat([tmp_mult, mult[o]], 0)
            else:
                tmp = output[o]
                tmp_mult = mult[o]
            if tmp.shape[0] < batch_size:
                continue
            else:
                tmp = tmp / batch_size
                tmp_mult = tmp_mult / batch_size
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += tmp * tmp_mult
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += tmp_mult
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += tmp * tmp_mult
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += tmp_mult
            tmp = None
            tmp_mult = None
        del mult
        # tidy_up_cond_or_unconds = []
        # tidy_up_outputs = []
        # tidy_up_mults = []
        # tidy_up_areas = []  # (height, width, y, x)
        # cond_output_tmp, uncond_output_tmp = None, None
        # cond_mult_tmp, uncond_mult_tmp = None, None
        # cond_area_tmp, uncond_area_tmp = None, None
        # for i in range(len(output)):
        #     if cond_or_uncond[i] == COND:
        #         if cond_output_tmp is None:
        #             cond_output_tmp = output[i]
        #             cond_mult_tmp = mult[i]
        #             cond_area_tmp = area[i]
        #             print('cond_area_tmp:', cond_area_tmp)
        #             print('cond_mult_tmp:', cond_mult_tmp.shape)
        #             print('cond_output_tmp:', cond_output_tmp.shape)
        #             print(cond_output_tmp.max(), cond_output_tmp.min(), cond_output_tmp.mean(), cond_output_tmp.std(), cond_output_tmp.sum())
        #         else:
        #             cond_output_tmp = torch.cat([cond_output_tmp.clone(), output[i].clone()], 0).contiguous()
        #             cond_mult_tmp = torch.cat([cond_mult_tmp, mult[i]], 0)
        #             cond_area_tmp = (min((cond_area_tmp[0] + area[i][0]) // 2,  input_latent_shape[0]), # height
        #                              min((cond_area_tmp[1] + area[i][1]) // 2, input_latent_shape[1]),  # width
        #                              max((cond_area_tmp[2] + area[i][2]) // 2, 0),  # y
        #                              max((cond_area_tmp[3] + area[i][3]) // 2, 0))  # x
        #         if cond_output_tmp is not None and cond_output_tmp.shape[0] == batch_size:
        #             tidy_up_cond_or_unconds.append(COND)
        #             tidy_up_outputs.append(cond_output_tmp)
        #             tidy_up_mults.append(cond_mult_tmp)
        #             tidy_up_areas.append(cond_area_tmp)
        #             cond_output_tmp, cond_mult_tmp, cond_area_tmp = None, None, None
        #     else:
        #         if uncond_output_tmp is None:
        #             uncond_output_tmp = output[i]
        #             uncond_mult_tmp = mult[i]
        #             uncond_area_tmp = area[i]
        #         else:
        #             uncond_output_tmp = torch.cat([uncond_output_tmp.clone(), output[i].clone()], 0).contiguous()
        #             uncond_mult_tmp = torch.cat([uncond_mult_tmp, mult[i]], 0)
        #             uncond_area_tmp = (min((uncond_area_tmp[0] + area[i][0]) // 2,  input_latent_shape[0]), # height
        #                                min((uncond_area_tmp[1] + area[i][1]) // 2, input_latent_shape[1]),  # width
        #                                max((uncond_area_tmp[2] + area[i][2]) // 2, 0),  # y
        #                                max((uncond_area_tmp[3] + area[i][3]) // 2, 0))  # x
        #         if uncond_output_tmp is not None and uncond_output_tmp.shape[0] == batch_size:
        #             tidy_up_cond_or_unconds.append(UNCOND)
        #             tidy_up_outputs.append(uncond_output_tmp)
        #             tidy_up_mults.append(uncond_mult_tmp)
        #             tidy_up_areas.append(uncond_area_tmp)
        #             uncond_output_tmp, uncond_mult_tmp, uncond_area_tmp = None, None, None
        
        # for o in range(len(tidy_up_cond_or_unconds)):
        #     print(o, 'cond' if tidy_up_cond_or_unconds[o] == COND else 'uncond', tidy_up_outputs[o].shape, tidy_up_mults[o].shape, tidy_up_areas[o])
        #     print(tidy_up_mults[o].max(), tidy_up_mults[o].min(), tidy_up_mults[o].mean(), tidy_up_mults[o].std(), tidy_up_mults[o].sum())
        #     print(tidy_up_outputs[o].max(), tidy_up_outputs[o].min(), tidy_up_outputs[o].mean(), tidy_up_outputs[o].std(), tidy_up_outputs[o].sum())
        #     if tidy_up_cond_or_unconds[o] == COND:
        #         out_cond[:,:,tidy_up_areas[o][2]:tidy_up_areas[o][0] + tidy_up_areas[o][2],tidy_up_areas[o][3]:tidy_up_areas[o][1] + tidy_up_areas[o][3]] += tidy_up_outputs[o] * tidy_up_mults[o]
        #         out_count[:,:,tidy_up_areas[o][2]:tidy_up_areas[o][0] + tidy_up_areas[o][2],tidy_up_areas[o][3]:tidy_up_areas[o][1] + tidy_up_areas[o][3]] += tidy_up_mults[o]
        #     else:
        #         out_uncond[:,:,tidy_up_areas[o][2]:tidy_up_areas[o][0] + tidy_up_areas[o][2],tidy_up_areas[o][3]:tidy_up_areas[o][1] + tidy_up_areas[o][3]] += tidy_up_outputs[o] * tidy_up_mults[o]
        #         out_uncond_count[:,:,tidy_up_areas[o][2]:tidy_up_areas[o][0] + tidy_up_areas[o][2],tidy_up_areas[o][3]:tidy_up_areas[o][1] + tidy_up_areas[o][3]] += tidy_up_mults[o]
        # del mult

    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond