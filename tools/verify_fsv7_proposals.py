"""Regression and integration checks for the three independent FSV7 proposals.

Run from the repository root:
    python tools/verify_fsv7_proposals.py -v
No dataset, weights, downloads, or training run directories are required.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.extra_modules.aifi_block_sparse import AIFI_BlockSparse
from ultralytics.nn.extra_modules.aifi_slgm import AIFI_SLGM, AIFI_BlockSparseSLGM
from ultralytics.nn.extra_modules.decoder_cir import ContentAwareBilinear, MSDeformAttnCIR, RTDETRDecoderCIR
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import AIFI, MSDeformAttn
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.loss_plugins import build_loss
from ultralytics.utils.torch_utils import ModelEMA

CONFIG = ROOT / 'ultralytics/cfg/models/rt-detr'
PREFIX = 'rtdetr-r18-DSAWACGAv5-P2SPDOKMFSV7'
torch.set_num_threads(2)


class ProposalChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)

    def test_global_aifi_zero_and_active(self):
        for pre in (False, True):
            with self.subTest(pre=pre):
                base = AIFI(32,64,4,normalize_before=pre).double()
                adapted = AIFI_SLGM(32,64,4,normalize_before=pre).double()
                result = adapted.load_state_dict(base.state_dict(), strict=False)
                self.assertEqual(result.missing_keys, ['slgm_strength'])
                self.assertEqual(sum(p.numel() for p in base.parameters()), sum(p.numel() for p in adapted.parameters()))
                x = torch.randn(2,32,5,7,dtype=torch.float64,requires_grad=True)
                expected, actual = base(x), adapted(x)
                torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
                probe = torch.randn_like(expected)
                grad_base = torch.autograd.grad((expected*probe).sum(),x,retain_graph=True)[0]
                grad_adapted = torch.autograd.grad((actual*probe).sum(),x)[0]
                torch.testing.assert_close(grad_base,grad_adapted,atol=1e-10,rtol=1e-9)
                adapted.set_epoch(30,350)
                active = adapted(x)
                self.assertFalse(torch.allclose(active,expected,atol=1e-8,rtol=1e-8))
                (active*probe).sum().backward()
                self.assertTrue(torch.isfinite(adapted.ma.in_proj_weight.grad).all())
                self.assertGreater(adapted.ma.in_proj_weight.grad.abs().sum().item(),0)

    def test_optional_sparse_aifi_equivalence_and_padding(self):
        for pre in (False,True):
            with self.subTest(pre=pre):
                base = AIFI_BlockSparse(32,64,4,normalize_before=pre).double()
                adapted = AIFI_BlockSparseSLGM(32,64,4,normalize_before=pre).double()
                adapted.load_state_dict(base.state_dict(),strict=False)
                x = torch.randn(2,32,5,7,dtype=torch.float64,requires_grad=True)  # 35 tokens: requires padding
                expected, actual = base(x), adapted(x)
                torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
                adapted.set_epoch(30,350)
                active = adapted(x)
                self.assertTrue(torch.isfinite(active).all())
                self.assertFalse(torch.allclose(active,expected,atol=1e-8,rtol=1e-8))
                (active*torch.randn_like(active)).sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_cir_bilinear_boundary_values_and_coordinate_gradients(self):
        sampler = ContentAwareBilinear(32).double()
        self.assertEqual(sum(p.numel() for p in sampler.parameters()),945)
        feature = torch.randn(4,32,5,7,dtype=torch.float64,requires_grad=True)
        location = (torch.rand(4,11,4,2,dtype=torch.float64)*1.6-.3).requires_grad_()
        query = torch.randn(4,11,32,dtype=torch.float64)
        actual = sampler(feature,query,location,torch.tensor(0.,dtype=torch.float64))
        expected = F.grid_sample(feature,2*location-1,mode='bilinear',padding_mode='zeros',align_corners=False)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)
        probe = torch.randn_like(expected)
        ga = torch.autograd.grad((actual*probe).sum(),(feature,location),retain_graph=True)
        gb = torch.autograd.grad((expected*probe).sum(),(feature,location))
        for a,b in zip(ga,gb):
            torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_cir_constant_preservation_and_live_parameters(self):
        sampler = ContentAwareBilinear(32).double()
        location = torch.rand(2,7,4,2,dtype=torch.float64)*.5+.25
        query = torch.randn(2,7,32,dtype=torch.float64,requires_grad=True)
        constant = torch.full((2,32,5,7),3.,dtype=torch.float64)
        torch.testing.assert_close(sampler(constant,query,location,.5),torch.full((2,32,7,4),3.,dtype=torch.float64))
        feature = torch.randn_like(constant,requires_grad=True)
        output = sampler(feature,query,location,.5)
        (output*torch.randn_like(output)).sum().backward()
        for p in sampler.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(),0)

    def test_cir_deformable_attention_and_dn_isolation(self):
        for coords in (2,4):
            with self.subTest(coords=coords):
                base = MSDeformAttn(32,3,2,4).double()
                adapted = MSDeformAttnCIR(32,3,2,4,matching_queries=7).double()
                adapted.load_state_dict(base.state_dict(),strict=False)
                query = torch.randn(2,9,32,dtype=torch.float64)  # DN=2, matching=7
                value = torch.randn(2,51,32,dtype=torch.float64)
                shapes = [[5,7],[3,4],[2,2]]
                reference = torch.rand(2,9,1,coords,dtype=torch.float64)*.6+.2
                mask = torch.zeros(2,51,dtype=torch.bool); mask[:,0] = True
                expected = base(query,reference,value,shapes,mask)
                torch.testing.assert_close(adapted(query,reference,value,shapes,mask),expected,atol=1e-11,rtol=1e-10)
                adapted.set_epoch(30,350)
                output = adapted(query,reference,value,shapes,mask)
                torch.testing.assert_close(output[:,:2],expected[:,:2],atol=1e-12,rtol=1e-11)
                self.assertFalse(torch.allclose(output[:,2:],expected[:,2:]))

    def test_cir_full_head_dn_and_final_layer_only(self):
        base = RTDETRDecoder(3,(16,16,16),hd=32,nq=7,nh=2,ndl=3,nd=4)
        adapted = RTDETRDecoderCIR(3,(16,16,16),hd=32,nq=7,nh=2,ndl=3,nd=4)
        adapted.load_state_dict(base.state_dict(),strict=False)
        adapted.decoder.layers[-1].cross_attn.set_epoch(30,350)
        self.assertIs(type(adapted.decoder.layers[0].cross_attn),MSDeformAttn)
        features = [torch.randn(2,16,s,s) for s in (5,3,2)]
        batch = {'cls':torch.tensor([0,1,2]),'bboxes':torch.tensor([[.3,.3,.1,.1],[.4,.6,.2,.15],[.7,.7,.1,.1]]),
                 'gt_groups':[1,2],'batch_idx':torch.tensor([0,1,1])}
        torch.manual_seed(42); expected = base(features,batch)
        torch.manual_seed(42); actual = adapted(features,batch)
        dn = actual[4]['dn_num_split'][0]
        self.assertGreater(dn,0)
        torch.testing.assert_close(actual[0][:,:,:dn],expected[0][:,:,:dn])
        torch.testing.assert_close(actual[1][:,:,:dn],expected[1][:,:,:dn])
        torch.testing.assert_close(actual[1][:-1],expected[1][:-1])
        self.assertFalse(torch.allclose(actual[1][-1,:,dn:],expected[1][-1,:,dn:]))

    @staticmethod
    def loss_inputs():
        boxes = torch.rand(4,2,7,4).requires_grad_()
        scores = torch.randn(4,2,7,3,requires_grad=True)
        dn_boxes = torch.rand(3,2,4,4).requires_grad_()
        dn_scores = torch.randn(3,2,4,3,requires_grad=True)
        batch = {'cls':torch.tensor([0,1,2]), 'bboxes':torch.tensor([[.3,.3,.1,.1],[.4,.6,.2,.15],[.7,.7,.1,.1]]),
                 'gt_groups':[2,1]}
        dn_meta = {'dn_pos_idx':[torch.tensor([0,2]),torch.tensor([0])],'dn_num_group':1,'dn_num_split':[4,7]}
        return (boxes,scores),batch,dict(dn_bboxes=dn_boxes,dn_scores=dn_scores,dn_meta=dn_meta)

    def test_bqg_zero_equivalence_including_aux_and_dn(self):
        base = RTDETRDetectionLoss(nc=3,use_vfl=True)
        adapted = build_loss('bqg_vfl',3,{'lambda_max':0.,'warmup_epochs':0})
        preds,batch,dn = self.loss_inputs()
        expected,actual = base(preds,batch,**dn),adapted(preds,batch,**dn)
        self.assertEqual(expected.keys(),actual.keys())
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        grads_a = torch.autograd.grad(sum(actual.values()),(preds[1],dn['dn_scores']),retain_graph=True)
        grads_b = torch.autograd.grad(sum(expected.values()),(preds[1],dn['dn_scores']))
        for a,b in zip(grads_a,grads_b):
            torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_bqg_active_only_final_matching_and_one_ema_update(self):
        base = RTDETRDetectionLoss(nc=3,use_vfl=True)
        adapted = build_loss('bqg_vfl',3,{'warmup_epochs':0})
        preds,batch,dn = self.loss_inputs()
        expected = base(preds,batch,**dn)
        with patch.object(adapted,'_row_weights',wraps=adapted._row_weights) as spy:
            actual = adapted(preds,batch,**dn)
            self.assertEqual(spy.call_count,1)
        for key in expected:
            if key != 'loss_class':
                torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
        self.assertNotEqual(actual['loss_class'].item(),expected['loss_class'].item())

    def test_bqg_detached_row_gradient_scaling_and_empty_fallback(self):
        criterion = build_loss('bqg_vfl',3,{'warmup_epochs':0}).eval()
        base = RTDETRDetectionLoss(nc=3,use_vfl=True)
        scores = torch.randn(2,7,3,requires_grad=True)
        targets = torch.full((2,7),3,dtype=torch.long)
        targets[0,0],targets[0,1],targets[1,3] = 0,1,2
        quality = torch.zeros(2,7);quality[0,0],quality[0,1],quality[1,3] = .7,.3,.5
        weights = criterion._row_weights(scores,targets,quality)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights[targets<3].mean(),torch.tensor(1.))
        self.assertTrue(torch.equal(weights[targets==3],torch.ones_like(weights[targets==3])))
        a = criterion._get_loss_class(scores,targets,quality,3)['loss_class']
        b = base._get_loss_class(scores,targets,quality,3)['loss_class']
        ga = torch.autograd.grad(a,scores,retain_graph=True)[0]
        gb = torch.autograd.grad(b,scores)[0]
        torch.testing.assert_close(ga,weights[...,None]*gb,atol=2e-7,rtol=2e-6)
        empty_targets,empty_q = torch.full_like(targets,3),torch.zeros_like(quality)
        criterion.train()
        before = deepcopy(criterion.state_dict())
        ea = criterion._get_loss_class(scores,empty_targets,empty_q,0)['loss_class']
        eb = base._get_loss_class(scores,empty_targets,empty_q,0)['loss_class']
        torch.testing.assert_close(ea,eb,atol=0,rtol=0)
        for key in before:
            torch.testing.assert_close(criterion.state_dict()[key],before[key],atol=0,rtol=0)

    def test_schedule_checkpoint_and_eval_statistics(self):
        objects = [AIFI_SLGM(32,64,4),MSDeformAttnCIR(32,3,2,matching_queries=7),build_loss('bqg_vfl',3)]
        buffers,maxima = ['slgm_strength','cir_strength','bqg_strength'],[.25,.5,.25]
        for obj,buf,maximum in zip(objects,buffers,maxima):
            obj.set_epoch(15,350)
            self.assertAlmostEqual(getattr(obj,buf).item(),maximum/2)
            copy = deepcopy(obj);copy.set_epoch(0,350);copy.load_state_dict(obj.state_dict())
            self.assertEqual(getattr(copy,buf).item(),getattr(obj,buf).item())
        criterion = objects[-1]
        scores = torch.randn(1,2,3);targets = torch.tensor([[0,3]]);q = torch.tensor([[.6,0.]])
        criterion._get_loss_class(scores,targets,q,1)
        state = deepcopy(criterion.state_dict());criterion.eval()
        criterion._get_loss_class(scores+1,targets,q,1)
        for key in state:
            torch.testing.assert_close(criterion.state_dict()[key],state[key],atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_amp_forward_backward(self):
        aifi = AIFI_SLGM(32,64,4).cuda()
        cross = MSDeformAttnCIR(32,3,2,matching_queries=7).cuda()
        criterion = build_loss('bqg_vfl',3).cuda()
        for obj in (aifi,cross,criterion):
            obj.set_epoch(30,350)
        image = torch.randn(2,32,5,7,device='cuda',requires_grad=True)
        query = torch.randn(2,7,32,device='cuda',requires_grad=True)
        values = torch.randn(2,51,32,device='cuda',requires_grad=True)
        reference = torch.rand(2,7,1,4,device='cuda',requires_grad=True)
        scores = torch.randn(2,7,3,device='cuda',requires_grad=True)
        targets = torch.full((2,7),3,dtype=torch.long,device='cuda');targets[0,0] = 1
        quality = torch.zeros(2,7,device='cuda');quality[0,0] = .6
        with torch.cuda.amp.autocast():
            output = aifi(image)
            sampled = cross(query,reference,values,[[5,7],[3,4],[2,2]])
            loss = output.float().square().mean() + sampled.float().square().mean()
            loss = loss + criterion._get_loss_class(scores,targets,quality,1)['loss_class']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for tensor in (image,query,values,reference,scores):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for p in cross.resampler.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_configs_full_model_forward_backward_and_ema(self):
        base_cfg = yaml.safe_load((CONFIG/(PREFIX+'.yaml')).read_text(encoding='utf-8'))
        suffixes = ['-AIFI-SLGM','-AIFI-BlockSparse-SLGM','-Decoder-CIR','-BQG-VFL']
        base_model = RTDETRDetectionModel(str(CONFIG/(PREFIX+'.yaml')),verbose=False)
        base_parameters = sum(p.numel() for p in base_model.parameters())
        del base_model
        for suffix in suffixes:
            with self.subTest(suffix=suffix):
                cfg_path = CONFIG/(PREFIX+suffix+'.yaml')
                cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
                self.assertEqual(cfg['backbone'],base_cfg['backbone'])
                for i,(before,after) in enumerate(zip(base_cfg['head'],cfg['head'])):
                    if suffix.startswith('-AIFI') and i==1 or suffix=='-Decoder-CIR' and i==15:
                        continue
                    self.assertEqual(before,after)
                model = RTDETRDetectionModel(str(cfg_path),verbose=False)
                self.assertEqual(sum(p.numel() for p in model.parameters())-base_parameters,945 if suffix=='-Decoder-CIR' else 0)
                if suffix in ('-Decoder-CIR','-BQG-VFL'):
                    self.assertIs(type(model.model[10]),AIFI)
                    self.assertFalse(any(isinstance(m,AIFI_BlockSparse) for m in model.modules()))
                ema = ModelEMA(model) if suffix=='-BQG-VFL' else None
                model.train()
                for module in model.modules():
                    if getattr(module,'innovation_schedule',False):
                        module.set_epoch(30,350)
                batch = {'img':torch.randn(2,3,128,128),'cls':torch.tensor([[3],[0],[1]]),
                         'bboxes':torch.tensor([[.3,.3,.1,.1],[.4,.6,.2,.15],[.7,.7,.1,.1]]),
                         'batch_idx':torch.tensor([0,1,1]),'_innovation_epoch':30,'_innovation_epochs':350}
                loss,items = model(batch)
                self.assertTrue(torch.isfinite(loss).all())
                self.assertTrue(torch.isfinite(items).all())
                loss.backward()
                if suffix=='-Decoder-CIR':
                    for p in model.model[-1].decoder.layers[-1].cross_attn.resampler.parameters():
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertGreater(p.grad.abs().sum().item(),0)
                if ema is not None:
                    ema.update(model)
                    self.assertIn('criterion.bqg_error_ema',ema.ema.state_dict())
                    self.assertGreater(ema.ema.criterion.bqg_initialized.item(),.5)
                model.eval()
                with torch.no_grad():
                    prediction = model(batch['img'])[0]
                self.assertEqual(tuple(prediction.shape),(2,300,14))
                self.assertTrue(torch.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
