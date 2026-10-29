import csv, importlib, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fire2025_reproduction as r
import run_hpc

class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.smoke=self.root/'smoke'; self.dfire=self.root/'dfire'; self.out=self.root/'output'
        for dataset,root,structure in [('s',self.smoke,{'train':6,'valid':2,'test':2}),('d',self.dfire,{'train':8,'test':2})]:
            for split,number in structure.items():
                im=root/split/'images'; lab=root/split/'labels'; im.mkdir(parents=True);lab.mkdir()
                for i in range(number):
                    name=f'{dataset}_{split}_{i:03d}'
                    (im/(name+'.jpg')).write_bytes(b'fake-image')
                    label='' if i==0 else '0 0.5 0.5 0.2 0.2\n'
                    (lab/(name+'.txt')).write_text(label)
    def tearDown(self):self.temp.cleanup()
    def prepare(self):
        return r.prepare_datasets(self.smoke,self.dfire,self.out,['fire','smoke'],
                                  dfire_val_count=2,split_seed=42,expected_counts=False)
    def test_experiment_matrix(self):
        jobs=r.build_experiments(); self.assertEqual(len(jobs),13)
        self.assertEqual(len({j['id'] for j in jobs}),13)
        self.assertEqual({j['model'] for j in jobs},set(r.ALL_MODELS))
        self.assertEqual(sum(j['epochs'] for j in jobs),1550)
        for j in jobs:
            self.assertIn(j['id'],r.PAPER_SCORES)
            if j['id'].startswith('t7_smoke'):self.assertEqual((j['train_args']['imgsz'],j['train_args']['batch']),(416,64))
            if j['id'].startswith('t4_'):self.assertEqual((j['train_args']['optimizer'],j['train_args']['batch']),('Adam',16))
        self.assertEqual(r.build_experiments({'batch':32})[-1]['train_args']['batch'],32)
    def test_prepare_and_reuse_deterministic_manifests(self):
        before={p:p.read_bytes() for root in (self.smoke,self.dfire) for p in root.rglob('*') if p.is_file()}
        a=self.prepare();b=self.prepare();self.assertEqual(a,b)
        self.assertEqual(a['datasets']['dfire']['split_sizes'],{'train':6,'val':2,'test':2})
        self.assertFalse((self.dfire/'val').exists())
        self.assertEqual(a['audit_original_splits']['dfire/train']['negative_images'],1)
        self.assertEqual(a['audit_original_splits']['dfire/test']['negative_images'],1)
        self.assertNotIn('dfire/val',a['audit_original_splits'])
        for ds in ('smoke_fire','dfire'):
            yaml_path=Path(a['datasets'][ds]['yaml'])
            self.assertTrue(yaml_path.is_relative_to(self.out))
            data=json.loads(yaml_path.read_text())
            sets=[set(Path(data[s]).read_text().splitlines()) for s in ('train','val','test')]
            self.assertTrue(all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i)))
            self.assertEqual(data['names'],a['datasets'][ds]['names'])
            for split,paths in zip(('train','val','test'),sets):
                self.assertTrue(Path(data[split]).is_absolute())
                self.assertTrue(Path(data[split]).is_relative_to(self.out))
                self.assertEqual(len(paths),a['datasets'][ds]['split_sizes'][split])
                for line in paths:
                    image=Path(line)
                    self.assertTrue(image.is_absolute());self.assertIn(image,before)
                    expected_label=image.parent.parent/'labels'/f'{image.stem}.txt'
                    self.assertEqual(r.label_for_image(image),expected_label)
                    self.assertIn(expected_label,before)
            source_root=self.smoke if ds=='smoke_fire' else self.dfire
            self.assertEqual(sets[2],{str(p.resolve()) for p in (source_root/'test/images').glob('*.jpg')})
            if ds=='dfire':
                self.assertEqual(sets[0]|sets[1],{str(p.resolve()) for p in (self.dfire/'train/images').glob('*.jpg')})
            else:
                for split,paths in zip(('train','valid'),sets[:2]):
                    self.assertEqual(paths,{str(p.resolve()) for p in (self.smoke/split/'images').glob('*.jpg')})
        after={p:p.read_bytes() for root in (self.smoke,self.dfire) for p in root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual({p.relative_to(self.out).as_posix() for p in self.out.rglob('*') if p.is_file()},
                         {'data_manifests/provenance.json'} |
                         {f'data_manifests/{ds}/{name}' for ds in ('smoke_fire','dfire')
                          for name in ('train.txt','val.txt','test.txt','data.yaml')})
        with self.assertRaises(FileExistsError):r.prepare_datasets(self.smoke,self.dfire,self.out,['fire','smoke'],dfire_val_count=2,split_seed=33,expected_counts=False)

    def test_official_counts_and_seed_are_deterministic(self):
        # Virtual synthetic paths exercise full counts; the small fixture tests real label I/O.
        counts={self.smoke:{'train':8494,'valid':2114,'test':1059},
                self.dfire:{'train':17221,'test':4306}}
        sources={root/split/'images':[root/split/'images'/f'{split}_{i:05d}.jpg' for i in range(number)]
                 for root,splits in counts.items() for split,number in splits.items()}
        def clean_audit(images):
            return {'images':len(images),'missing_labels':0,'unreadable_labels':0,
                    'malformed_rows':0,'invalid_values':0}
        with patch.object(r,'image_paths',side_effect=lambda path:sources[path]), \
             patch.object(r,'audit_split',side_effect=clean_audit):
            a=r.prepare_datasets(self.smoke,self.dfire,self.out,['fire','smoke'])
            b=r.prepare_datasets(self.smoke,self.dfire,self.root/'repeat',['fire','smoke'])
            self.assertEqual(a['datasets']['dfire']['split_sizes'],{'train':15498,'val':1723,'test':4306})
            self.assertEqual(a['dfire_split_seed'],42)
            self.assertEqual(a['datasets']['dfire']['list_sha256'],b['datasets']['dfire']['list_sha256'])
            data=json.loads(Path(a['datasets']['dfire']['yaml']).read_text())
            splits={split:Path(data[split]).read_text().splitlines() for split in ('train','val','test')}
            train,val,test=(set(splits[split]) for split in ('train','val','test'))
            for split,paths in splits.items():
                self.assertEqual(len(paths),a['datasets']['dfire']['split_sizes'][split])
                self.assertEqual(len(paths),len(set(paths)))
            self.assertTrue(train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test))
            self.assertEqual(train|val,{str(p) for p in sources[self.dfire/'train/images']})
            self.assertEqual(test,{str(p) for p in sources[self.dfire/'test/images']})
            sources[self.dfire/'train/images'].pop()
            with self.assertRaisesRegex(ValueError,'dfire/train: ditemukan 17220, diharapkan 17221'):
                r.prepare_datasets(self.smoke,self.dfire,self.root/'wrong-count',['fire','smoke'])
            self.assertFalse((self.root/'wrong-count').exists())

    def test_legacy_dfire_layout_remains_supported(self):
        data=self.dfire/'data';data.mkdir()
        for split in ('train','test'):
            (self.dfire/split).rename(data/split)
        for folder in ('images','labels'):
            (data/'val'/folder).mkdir(parents=True,exist_ok=True)
        for image in sorted((data/'train/images').glob('*.jpg'))[-2:]:
            label=r.label_for_image(image)
            image.rename(data/'val/images'/image.name)
            label.rename(data/'val/labels'/label.name)
        result=self.prepare()
        self.assertEqual(result['datasets']['dfire']['split_sizes'],{'train':6,'val':2,'test':2})
        self.assertEqual(result['audit_original_splits']['dfire/val']['images'],2)

    def test_source_overlap_and_output_inside_source_fail(self):
        for output in (self.smoke,self.smoke/'output',self.dfire,self.dfire/'output'):
            with self.subTest(output=output),self.assertRaisesRegex(ValueError,'OUTPUT_ROOT'):
                r.prepare_datasets(self.smoke,self.dfire,output,['fire','smoke'],expected_counts=False)
        image=next((self.dfire/'train/images').glob('*.jpg'))
        (self.dfire/'test/images'/image.name).write_bytes(image.read_bytes())
        label=r.label_for_image(image)
        (self.dfire/'test/labels'/label.name).write_bytes(label.read_bytes())
        with self.assertRaisesRegex(ValueError,'nama gambar sama antara train dan test'):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_audit_cli_needs_no_weights_or_training_environment(self):
        provenance=self.prepare()
        argv=['run_hpc.py','--smoke-root',str(self.smoke),'--dfire-root',str(self.dfire),
              '--output-root',str(self.out),'--smoke-classes','fire','smoke']
        with patch.object(sys,'argv',argv),patch.object(r,'prepare_datasets',return_value=provenance) as prepare, \
             patch.object(r,'require_training_environment',side_effect=AssertionError('GPU check during audit')), \
             patch.object(r,'validate_weight_files',side_effect=AssertionError('Weight check during audit')), \
             patch.object(r,'run_queue',side_effect=AssertionError('Training during audit')),patch('builtins.print'):
            run_hpc.main()
        prepare.assert_called_once_with(self.smoke,self.dfire,self.out,['fire','smoke'],['smoke','fire'],
                                        dfire_val_count=1723,split_seed=42)
        with patch.object(sys,'argv',argv+['--mode','train']),patch('sys.stderr'), \
             self.assertRaises(SystemExit) as error:
            run_hpc.main()
        self.assertEqual(error.exception.code,2)
    def test_label_fingerprint_detects_mutation(self):
        self.prepare()
        label = self.smoke / 'train' / 'labels' / 's_train_001.txt'
        label.write_text('0 0.5 0.5 0.3 0.2\n')
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_unknown_classes_and_missing_labels_fail(self):
        with self.assertRaises(ValueError):r.prepare_datasets(self.smoke,self.dfire,self.out,None,expected_counts=False)
        next((self.smoke/'train'/'labels').glob('*.txt')).unlink()
        with self.assertRaisesRegex(ValueError,'missing_labels'):self.prepare()
    def test_audit_counts_issues_without_editing_data(self):
        p=self.dfire/'test'/'labels'/'d_test_001.txt'
        data='0 0.5 0.5 0 0\n1 0.5 0.5 1.02 0.1\n';p.write_text(data)
        stats=r.audit_split(r.image_paths(self.dfire/'test'/'images'))
        self.assertEqual(stats['zero_area_boxes'],1);self.assertEqual(stats['oversized_boxes'],1)
        self.assertEqual(stats['outside_image_boxes'],1);self.assertEqual(p.read_text(),data)
    def test_queue_runs_once_resume_and_evaluation(self):
        prov=self.prepare();weights=self.root/'weights';weights.mkdir()
        (weights/'yolo11n.pt').write_bytes(b'initial-weights')
        job=next(x.copy() for x in r.build_experiments() if x['id']=='t4_yolo11n_50')
        job['epochs']=2; calls=[]
        class FakeModel:
            def __init__(self,path):
                self.path=Path(path);self.task="detect"
                self.model=SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 2)])
            def train(self,**kwargs):
                calls.append(('train',kwargs))
                if kwargs.get('resume'):
                    run=self.path.parent.parent
                else:run=Path(kwargs['project'])/kwargs['name']
                (run/'weights').mkdir(parents=True,exist_ok=True)
                (run/'weights'/'last.pt').write_bytes(b'last')
                (run/'weights'/'best.pt').write_bytes(b'best')
                (run/'results.csv').write_text('epoch,metric\n2,0.8\n')
            def val(self,**kwargs):
                calls.append(('val',kwargs));(Path(kwargs['project'])/kwargs['name']).mkdir(parents=True)
                box=SimpleNamespace(ap_class_index=[0,1],ap50=[.7,.8],ap=[.4,.5],p=[.6,.7],r=[.5,.6])
                return SimpleNamespace(results_dict={'metrics/precision(B)':.65,'metrics/recall(B)':.55,'metrics/mAP50(B)':.75,'metrics/mAP50-95(B)':.45},box=box)
        modules={'ultralytics':SimpleNamespace(YOLO=FakeModel),'ultralytics.utils':SimpleNamespace(AUTOINSTALL=True),'ultralytics.utils.checks':SimpleNamespace(AUTOINSTALL=True)}
        info={'python':'3','platform':'test','packages':{'torch':'test','ultralytics':'test'},'gpu':{'name':'A100'},'cuda_visible_devices':'3'}
        with patch.object(r,'require_training_environment',return_value=info),patch.object(r.importlib,'import_module',side_effect=lambda x:modules[x]):
            ans=r.run_queue([job],prov,weights,self.out,device=0,workers=1)
            self.assertEqual(ans[0]['status'],'complete');self.assertEqual(len(calls),3)
            self.assertEqual(calls[0][1]['device'],'');self.assertEqual(calls[1][1]['device'],'')
            self.assertFalse(modules['ultralytics.utils.checks'].AUTOINSTALL)
            self.assertEqual(ans[0]['metrics']['test']['per_class']['fire']['ap50'],.7)
            r.run_queue([job],prov,weights,self.out,device=0,workers=1);self.assertEqual(len(calls),3)
            csvpath,rows=r.summarize_results([job],self.out);self.assertTrue(csvpath.is_file());self.assertEqual(rows[0]['test_map50'],.75)
            # Simulate a failed run with a last.pt and an incomplete CSV, then resume it.
            (self.out/'metadata'/f"{job['id']}.complete.json").unlink()
            for split in ('val','test'):
                (self.out/'evaluation'/job['id']/split/'metrics.json').unlink()
                (self.out/'evaluation'/job['id']/split).rmdir()
            (self.out/'training'/job['id']/'results.csv').write_text('epoch,metric\n1,0.5\n')
            r.run_queue([job],prov,weights,self.out,device=0,workers=1)
            self.assertTrue(calls[3][1]['resume'])
            self.assertEqual(calls[3][1]['device'],'')
    def test_notebook_code_executes_without_data_or_training(self):
        nb=json.loads((ROOT/'Fire2025_Reproduction.ipynb').read_text())
        self.assertEqual(nb['nbformat'],4)
        ns={}
        old=os.getcwd()
        try:
            os.chdir(ROOT)
            for index,cell in enumerate(nb['cells']):
                if cell['cell_type']=='code':
                    code=''.join(cell['source']);compile(code,f'cell-{index}','exec');exec(code,ns)
        finally:os.chdir(old)

if __name__=='__main__':unittest.main(verbosity=2)
