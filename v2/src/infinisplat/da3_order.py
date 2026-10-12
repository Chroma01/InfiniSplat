# Copyright (c) Meta Platforms, Inc. and affiliates.
# Inference adaptation under Apache-2.0; see LICENSE_DA3.
# Modification: restore the input view order for exported appearance features.
import torch
from depth_anything_3.model.reference_view_selector import select_reference_view, reorder_by_reference, restore_original_order
from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION
from depth_anything_3.utils.logger import logger

def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
    B, S, _, H, W = x.shape
    output, total_block_len, aux_output = ([], len(self.blocks), [])
    blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
    cache = kwargs.get('prefix_cache')
    start = 0
    if kwargs.get('prefix_resume', False):
        # Cache belongs to this forward's two branches, never to another request.
        x, local_x = cache['x'].clone(), cache['local_x'].clone()
        pos, pos_nodiff = cache['pos'], cache['pos_nodiff']
        start = cache['blocks']
        output = list(cache['output'])
        if cache['reference'] is not None:
            b_idx = cache['reference']
    else:
        x = self.prepare_tokens_with_masks(x)
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)
    for i, blk in enumerate(self.blocks):
        if i < start:
            continue
        if i < self.rope_start or self.rope is None:
            g_pos, l_pos = (None, None)
        else:
            g_pos = pos_nodiff
            l_pos = pos
        if self.alt_start != -1 and i == self.alt_start - 1 and (x.shape[1] >= THRESH_FOR_REF_SELECTION) and (kwargs.get('cam_token', None) is None):
            strategy = kwargs.get('ref_view_strategy', 'saddle_balanced')
            logger.info(f'Selecting reference view using strategy: {strategy}')
            b_idx = select_reference_view(x, strategy=strategy)
            x = reorder_by_reference(x, b_idx)
            local_x = reorder_by_reference(local_x, b_idx)
        if self.alt_start != -1 and i == self.alt_start:
            if kwargs.get('cam_token', None) is not None:
                logger.info('Using camera conditions provided by the user')
                cam_token = kwargs.get('cam_token')
            else:
                ref_token = self.camera_token[:, :1].expand(B, -1, -1)
                src_token = self.camera_token[:, 1:].expand(B, S - 1, -1)
                cam_token = torch.cat([ref_token, src_token], dim=1)
            x[:, :, 0] = cam_token
        if self.alt_start != -1 and i >= self.alt_start and (i % 2 == 1):
            x = self.process_attention(x, blk, 'global', pos=g_pos, attn_mask=kwargs.get('attn_mask', None))
        else:
            x = self.process_attention(x, blk, 'local', pos=l_pos)
            local_x = x
        if i in blocks_to_take:
            out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
            if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and ('b_idx' in locals()):
                out_x = restore_original_order(out_x, b_idx)
            output.append((out_x[:, :, 0], out_x))
        if i in export_feat_layers:
            aux_x = x
            if x.shape[1] >= THRESH_FOR_REF_SELECTION and self.alt_start != -1 and ('b_idx' in locals()):
                aux_x = restore_original_order(aux_x, b_idx)
            aux_output.append(aux_x)
        if cache is not None and not kwargs.get('prefix_resume', False) and i + 1 == kwargs['prefix_blocks']:
            cache.update(x=x, local_x=local_x, pos=pos, pos_nodiff=pos_nodiff,
                         blocks=i+1, output=list(output), reference=locals().get('b_idx'))
    return (output, aux_output)
