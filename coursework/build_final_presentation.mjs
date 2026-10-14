import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import crypto from 'node:crypto';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=path.join(process.env.HOME,'.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill=path.join(process.env.HOME,'.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations');
process.env.RUNTIME_NODE_MODULES??=path.join(runtime,'node/node_modules');
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const base=path.join(root,'runs/materials_final_20261014');
await fs.mkdir(base,{recursive:true});
const historical=path.join(root,'runs/hqs_ablation_20261013');
const source=path.join(root,'output/presentation/course_presentation_ablation_20261013.pptx');
const facts=JSON.parse(await fs.readFile(path.join(historical,'report_facts.json'),'utf8'));
const oldBuild=JSON.parse(await fs.readFile(path.join(historical,'presentation_build.json'),'utf8'));
const recent=JSON.parse(await fs.readFile(path.join(root,'runs/hqs_20000_eval_20261014/results.json'),'utf8'));
const deck=await PresentationFile.importPptx(await FileBlob.load(source));
if(deck.slides.items.length!==12)throw new Error('Expected the twelve-slide source');
// This bundled runtime exposes the compatibility inspect overload.
const before=await deck.inspect({kind:'slide,layout,textbox,shape,image,table,chart,notes',maxChars:30000});
await fs.writeFile(path.join(base,'source_inspection.ndjson'),before.ndjson);
const build=await fs.mkdtemp(path.join(root,'tmp/presentation/final-'));
const output=path.join(root,'output/presentation',process.env.PRESENTATION_FILENAME??'course_presentation_final_20261014_v2.pptx');
const font='Helvetica Neue',ink='#20272B',teal='#147D78',muted='#59666C';
const times=[20,40,45,45,55,55,60,75,60,55,50,40];
if(times.reduce((a,b)=>a+b,0)!==600)throw new Error('Expected ten-minute speaking plan');
const notes={...oldBuild.notes};
const oldPair=(candidate,reference,sigma=0)=>facts.phases.primary.analysis.paired_summary.find(r=>r.dataset==='DIV2K_holdout_10000'&&r.cr===.1&&r.sigma===sigma&&r.candidate===candidate&&r.reference===reference);
const newPair=(candidate,reference,sigma=0,dataset='DIV2K_holdout_10000',phase='primary')=>recent.phases[phase].paired_summary.find(r=>r.dataset===dataset&&r.cr===.1&&r.sigma===sigma&&r.candidate===candidate&&r.reference===reference);
const factor=(comparison,seed,sigma)=>facts.phases.primary.analysis.paired_summary.find(r=>r.dataset==='DIV2K_holdout_10000'&&r.cr===.1&&r.sigma===sigma&&r.training_seed===seed&&r.checkpoint_view==='last'&&r.comparison===comparison);
function text(slide,value,x,y,w,h,size=28,bold=false,color=ink){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};return shape;
}
function reset(index,title,{keepImages=false,keepTables=false}={}){
  const slide=deck.slides.items[index-1];slide.shapes.deleteAll();
  for(const collection of [slide.charts,...(keepImages?[]:[slide.images]),...(keepTables?[]:[slide.tables])])
    for(const item of [...collection.items])collection.deleteById(item.id);
  slide.background.fill='#FFFFFF';
  text(slide,title,60,36,1155,100,44,true);
  text(slide,String(index).padStart(2,'0'),1180,673,50,24,17,false,muted);
  return slide;
}
function chart(slide,categories,series,yTitle){
  const c=slide.charts.add('bar',{position:{left:65,top:160,width:1135,height:420},categories,
    series:series.map(r=>({...r,values:r.values.map(v=>Number(v.toFixed(7)))})),
    hasLegend:true,legend:{position:'bottom',textStyle:{typeface:font,fontSize:21}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
    xAxis:{tickLabelPosition:'low',textStyle:{typeface:font,fontSize:20}},
    yAxis:{title:yTitle,numberFormatCode:'0.0%',textStyle:{typeface:font,fontSize:20},majorGridlines:{fill:'#E3E7E7',width:1}},
    barOptions:{direction:'column',grouping:'clustered'}});
  applyPresentationChartFont(c,{fontFamily:font});return c;
}

notes[1]='We simulate grayscale single-pixel reconstruction, compare four methods, fine-tune official HQS with frozen sensing factors, and run paired noise/stage-loss ablations with two training seeds. The main continuation completes 20000 total updates. Validation selects step 19500. The 20000 endpoint stays separate. On primary52, the extension slightly increases MSE relative to the prior selected step 11500, while SSIM rises slightly. Architecture and weights belong to Wang et al., CVPR 2025, github.com/pwangcs/ProxUnroll. The cover retains the scikit-image coins example. Codex assisted code, experiments, analysis and materials. Sources: technical_report_flow_20261014.pdf and both frozen experiment roots.';
let slide=reset(5,'Data splits and evaluation scenes',{keepImages:true});
text(slide,'800 training / 16 validation / 52 primary originals\nThe 20k extension rechecks 52 + 110 previously tested images',65,149,1120,88,28,true,teal);
for(const [i,label] of ['NJU Gulou','NJU Xianlin','Qinhuai night'].entries())text(slide,label,65+i*390,567,350,38,26);
text(slide,'External-author photographs; source and license retained',65,623,1120,35,23,false,muted);
notes[5]=oldBuild.notes[5].replace('The primary test is the previously locked, project-unused 52 DIV2K originals.','The 52 DIV2K originals were project-unused when locked for the previous experiment. The 20000 extension now reuses them and the 110 previously evaluated regression originals. This is a continuation check on observed test inputs. Test scores do not select weights or change the recipe. Original-source splits remain disjoint. Official pretraining exposure is unknown.');

// Rebuild native charts from complete frozen literal data. Importing an old
// chart alone can lose its embedded workbook. The finalizer materializes new
// workbook snapshots; the independent checker compares them to source CSV.
const baseline=JSON.parse(await fs.readFile(path.join(root,'runs/nanjing_web30_20261009/comparison/summary.json'),'utf8'));
slide=reset(6,'Four-method reconstruction on campus scenes');
const labels={adjoint:'Adjoint',fista_dct:'DCT-FISTA',hqs:'HQS pretrained',admm:'ADMM pretrained'};
const colors={adjoint:'#777777',fista_dct:'#D78924',hqs:teal,admm:'#B74760'};
const c=slide.charts.add('scatter',{position:{left:65,top:160,width:1135,height:420},categories:[],
  series:Object.entries(labels).map(([method,name])=>({name,xValues:[1,4,10,25,50],
    values:[.01,.04,.1,.25,.5].map(cr=>Number(baseline.datasets.find(r=>r.dataset==='NJU_campus_web'&&r.method===method&&r.cr===cr&&r.sigma===0).psnr.toFixed(5))),
    fill:colors[method],line:{fill:colors[method],width:3},marker:{symbol:'circle',size:7}})),
  hasLegend:true,legend:{position:'bottom',textStyle:{typeface:font,fontSize:21}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',scatterOptions:{style:'lineWithMarkers'},
  xAxis:{title:'Nominal clean sampling (%)',min:0,max:50,majorUnit:10,textStyle:{typeface:font,fontSize:20}},
  yAxis:{title:'PSNR (dB)',min:0,max:45,numberFormatCode:'0.00',textStyle:{typeface:font,fontSize:20},majorGridlines:{fill:'#E3E7E7',width:1}}});
applyPresentationChartFont(c,{fontFamily:font});
text(slide,'Every source photograph contributes equally',65,610,1120,45,25,false,muted);

slide=reset(7,'Matched training ablations',{keepTables:true});
text(slide,'Four arms, two seeds, 800 updates per arm\nSeparate full-state continuation: 10000 to 12000 to 20000',65,525,1120,91,28,true,teal);
text(slide,'All six sensing factors stay frozen',65,630,1120,34,23,false,muted);
notes[7]+=' A further authorized continuation restores the full 12000-update state, including Adam, scheduler, RNG and selection history, and adds 8000 unchanged-recipe updates to 20000. This takes 270.32 minutes and peaks at 10.84 GiB. The ablations retain their original 800-step budgets. The two continuations add 10000 updates, and eight ablation runs add 6400. Sources: runs/hqs_20000_20261013/finetune/{config,summary,audit}.json.';

slide=reset(8,'HQS continuation through 20000 updates');
chart(slide,['Clean','Noise .01','Noise .05'],[
  {name:'Selected 11500',fill:teal,values:[0,.01,.05].map(s=>oldPair('hqs_12000','hqs_10000',s).mse_reduction_percent/100)},
  {name:'Selected 19500',fill:'#B74760',values:[0,.01,.05].map(s=>newPair('hqs_20000','hqs_10000',s).mse_reduction_percent/100)},
  {name:'Endpoint 20000',fill:'#D78924',values:[0,.01,.05].map(s=>newPair('last_20000','hqs_10000',s).mse_reduction_percent/100)}
],'MSE reduction vs 10000-step HQS');
text(slide,'10% sampling on the same 52 originals\nValidation selects checkpoints. Positive bars mean lower error.',65,605,1120,58,23,false,muted);
notes[8]=`All three chart series share the selected 10000-step reference. At clean/noise .05, selected step 11500 reduces MSE by ${oldPair('hqs_12000','hqs_10000',0).mse_reduction_percent.toFixed(4)}% / ${oldPair('hqs_12000','hqs_10000',.05).mse_reduction_percent.toFixed(4)}%. Selected step 19500 instead gives ${newPair('hqs_20000','hqs_10000',0).mse_reduction_percent.toFixed(4)}% / ${newPair('hqs_20000','hqs_10000',.05).mse_reduction_percent.toFixed(4)}%. A negative MSE reduction means error rises. Relative to step 11500, selected 19500 increases MSE by ${(-newPair('hqs_20000','hqs_12000',0).mse_reduction_percent).toFixed(4)}% / ${(-newPair('hqs_20000','hqs_12000',.05).mse_reduction_percent).toFixed(4)}%. PSNR deltas are ${newPair('hqs_20000','hqs_12000',0).delta_psnr.toFixed(4)} / ${newPair('hqs_20000','hqs_12000',.05).delta_psnr.toFixed(4)} dB, and SSIM deltas are ${newPair('hqs_20000','hqs_12000',0).delta_ssim.toFixed(6)} / ${newPair('hqs_20000','hqs_12000',.05).delta_ssim.toFixed(6)}. The endpoint stays separate and also regresses on these primary MSE conditions. Initialization is an eligible validation candidate. No test result selects either state. These inputs were evaluated before this extension. Against official HQS, selected 19500 still reduces MSE by 3.3191% / 11.9766%, which describes accumulated adaptation, not the incremental benefit of the last 8000 updates. Sources: runs/hqs_20000_eval_20261014/evaluation_primary/paired_summary.csv and the frozen previous paired_summary.csv.`;

for(const [index,title,comparison,axis,footer] of [
  [9,'Effect of training measurement noise','noise_effect_at_weighted','MSE reduction: add training noise','Same 800-step endpoint; weighted stage loss held fixed'],
  [10,'Effect of intermediate-stage supervision','stage_effect_at_noise_on','MSE reduction: add stage supervision','Same 800-step endpoint; noisy training held fixed']]){
  slide=reset(index,title);
  chart(slide,['Clean','Noise .01','Noise .05'],[20261013,20261014].map((seed,i)=>({name:`Training seed ${seed}`,fill:[teal,'#D78924'][i],values:[0,.01,.05].map(s=>factor(comparison,seed,s).mse_reduction_percent/100)})),axis);
  text(slide,footer,65,610,1120,45,25,false,muted);
}

const failure=facts.failure_case;
slide=reset(11,'A preserved 11500-step failure',{keepImages:true});
text(slide,`11500 vs 10000: MSE reduction ${Number(failure.mse_reduction_percent).toFixed(2)}%\nSampling 4%; noise 0; image DIV2K 0897`,65,523,1120,84,28,true,teal);
text(slide,'Negative conditions: 191/572 at 11500; 258/572 at 19500',65,622,1120,35,23,false,muted);
notes[11]+=' The displayed illustration remains the validation-selected 11500 checkpoint, internally named hqs_12000. It is not a 19500 reconstruction. The 20000-budget selected state has 258/572 primary measurement conditions with higher MSE than step 11500. All negative cases and per-image deltas remain available. Source: runs/hqs_20000_eval_20261014/evaluation_primary/negative_cases.csv and paired_image_deltas.csv.';

slide=reset(12,'Results of the 20000-update extension');
text(slide,'Primary52 vs 11500: MSE rises 0.83% clean / 0.41% with noise .05',65,144,1120,47,28,true,teal);
text(slide,'SSIM rises slightly. Dataset changes remain small and mixed.',65,198,1120,40,26);
const datasetRows=[['Dataset','Clean','Noise .05']];
for(const [dataset,label] of [['Kodak','Kodak24'],['HEVC_B','HEVC B'],['HEVC_E','HEVC E'],['DIV2K_fresh','DIV2K32'],['NJU_campus_web','Campus18'],['Nanjing_scenic_web','Scenic12']]){
  datasetRows.push([label,...[0,.05].map(s=>{const v=newPair('hqs_20000','hqs_12000',s,dataset,'regression').mse_reduction_percent;return `${v>=0?'+':''}${v.toFixed(2)}%`;})]);
}
text(slide,'19500-step MSE reduction vs 11500, at 10% sampling',65,240,1120,30,23,false,muted);
const table=slide.tables.add({rows:datasetRows.length,columns:3,left:65,top:280,width:1135,height:252,columnWidths:[465,335,335],values:datasetRows});
for(let r=0;r<datasetRows.length;r++)for(let c=0;c<3;c++){
  const cell=table.getCell(r,c);cell.text.style={typeface:font,fontSize:23,bold:r===0,color:ink};cell.fill=r===0?'#EDF1F1':'#FFFFFF';
}
text(slide,'Software grayscale simulation. Video frames reconstruct independently.\nAI disclosure: Codex assisted code, experiments, analysis and materials.',65,612,1120,54,21,false,muted);
notes[12]='The selected 19500 checkpoint shows a small primary-MSE regression against selected 11500 and a small SSIM increase. Regression dataset changes are mixed and mostly fractions of a percent. Positive table values indicate lower MSE. Kodak remains separate from training and validation. Each HEVC video contributes equally, using three fixed spaced raw frames. We do not claim a further universal 5–10% improvement. The full comparison saves 3564 new final floating reconstructions and 21384 stage metrics over 162 originals. All new final metrics are independently recomputed, with identical per-condition measurements and source/operator hashes. Frozen official/10000/11500 scores pass a fresh reference-equivalence smoke check before reuse. The 800-step two-seed ablations remain distinct from the 20000 continuation. Sources: runs/hqs_20000_eval_20261014/{results,completion_audit}.json and evaluation_regression/paired_summary.csv. Limitations: modest tests, two ablation seeds, upstream pretraining exposure unknown, external-author photographs, grayscale software acquisition. Students must understand the code, rehearse and submit the coursework themselves.';

for(let i=1;i<=12;i++)deck.slides.items[i-1].speakerNotes.textFrame.setText(`Planned speaking time: ${times[i-1]} seconds.\n\n${notes[i]}`);
await fs.writeFile(path.join(root,'coursework/PRESENTATION_GUIDE_FINAL_20261014.md'),[
  '# Final Presentation Guide (2026-10-14)','Planned speaking time: 600 seconds. This twelve-slide version includes the completed 20000-update evaluation. Historical versions remain preserved.',
  ...Array.from({length:12},(_,i)=>`## Slide ${i+1} (${times[i]} seconds)\n\n${notes[i+1]}`)
].join('\n\n')+'\n');
const candidate=path.join(build,'candidate.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidate);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:output,
  pythonExecutable:path.join(runtime,'python/bin/python3'),
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','7','--require-native-table-slide','12'],
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[7,12],requiredNativeChartOwnerSlides:[6,8,9,10],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[font],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
  verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(output));
const previewDir=path.join(base,'slides');await fs.mkdir(previewDir,{recursive:true});
for(let i=0;i<12;i++){
  const blob=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});
  await fs.writeFile(path.join(previewDir,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
await fs.copyFile(path.join(build,'validation.json'),path.join(base,'presentation_validation.json'));
await fs.writeFile(path.join(base,'presentation_build.json'),JSON.stringify({output,previews:previewDir,notes,slideTimes:times,source,build},null,2)+'\n');
console.log(JSON.stringify({output,previews:previewDir,slides:12,speakingSeconds:600}));
