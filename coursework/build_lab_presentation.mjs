import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.CODEX_RUNTIME_ROOT??path.join(process.env.HOME,'.cache/codex-runtimes/codex-primary-runtime/dependencies');
const skill=process.env.PRESENTATIONS_SKILL_DIR??path.join(process.env.HOME,'.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations');
process.env.RUNTIME_NODE_MODULES??=path.join(runtime,'node/node_modules');
const {PresentationFile,FileBlob}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')));
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const source=path.join(root,'output/presentation/course_presentation.pptx');
const presentation=await PresentationFile.importPptx(await FileBlob.load(source));
const base=path.join(root,'runs/lab220_20261009');
const facts=JSON.parse(await fs.readFile(path.join(base,'report_facts.json'),'utf8'));
const {baseline,comparison,training,smoke,validation_curve:curve}=facts;
const output=path.join(root,'output/presentation',process.env.PRESENTATION_FILENAME??'course_presentation_lab220_20261009.pptx');
await fs.mkdir(path.join(root,'tmp/presentation'),{recursive:true});
const tmp=await fs.mkdtemp(path.join(root,'tmp/presentation/lab220-'));
const family='Helvetica Neue',ink='#20272B',teal='#147D78',muted='#59666C';
const times=[25,45,45,50,50,60,50,50,60,60,55,50];
const names={adjoint:'Adjoint',fista_dct:'DCT-FISTA',hqs:'HQS',admm:'ADMM'};
const colors={adjoint:'#777777',fista_dct:'#D78924',hqs:teal,admm:'#B74760'};
const datasets=['Kodak','HEVC_B','HEVC_E'];
const row=(d,m,cr=.1,sigma=0)=>baseline.datasets.find(r=>r.dataset===d&&r.method===m&&r.cr===cr&&r.sigma===sigma);
const delta=(d,cr=.1,m='hqs_finetuned')=>comparison.summary.find(r=>r.dataset===d&&r.method===m&&Number(r.cr)===cr&&Number(r.sigma)===0);
const guide=['# Lab220 Presentation Guide','Planned speaking time: 600 seconds. Original slides and reports remain preserved.'];
function text(s,value,x,y,w,h,size=28,bold=false,color=ink){const o=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});o.text=value;o.text.style={typeface:family,fontSize:size,bold,color,autoFit:'none'};return o;}
function reset(index,title,notes){const s=presentation.slides.items[index-1];s.shapes.deleteAll();for(const o of [...s.images.items])s.images.deleteById(o.id);for(const o of [...s.charts.items])s.charts.deleteById(o.id);s.background.fill='#FFFFFF';text(s,title,60,36,1155,100,44,true);text(s,String(index).padStart(2,'0'),1180,673,50,24,17,false,muted);s.speakerNotes.textFrame.setText(`Planned speaking time: ${times[index-1]} seconds.\n\n${notes}`);guide.push(`## Slide ${index}: ${title}`,notes);return s;}
function chart(s,type,categories,series,{x=65,y=160,w=1135,h=420,xTitle='',yTitle='PSNR (dB)'}={}){const c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,series:series.map(r=>({...r,values:r.values.map(v=>Number(v.toFixed(5)))})),hasLegend:true,legend:{position:'bottom',textStyle:{typeface:family,fontSize:21}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',xAxis:{title:xTitle,textStyle:{typeface:family,fontSize:20}},yAxis:{title:yTitle,numberFormatCode:'0.00',textStyle:{typeface:family,fontSize:20},majorGridlines:{fill:'#E3E7E7',width:1}},...(type==='bar'?{barOptions:{direction:'column',grouping:'clustered'}}:{lineOptions:{smooth:false}})});applyPresentationChartFont(c,{fontFamily:family});return c;}
async function image(s,filename,x,y,w,h,alt){s.images.add({blob:new Uint8Array(await fs.readFile(filename)),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
// Verified source ids from a read-only import inspection. Reuse the first four slides.
presentation.resolve('sh/k3yl0zql').text='External evaluation\nand fixed-operator fine-tuning';
presentation.resolve('sh/obq90bml').text='Official HQS and ADMM\nSix shared-weight stages\nMeasurement consistency updates\nFixed-operator HQS fine-tuning';
const introNotes=[
`We compare four solvers on public Kodak24 and spaced frames from eight laboratory-held HEVC videos. We fine-tune official HQS for ${training.completed_steps} bounded steps. Raw measurements and all test deltas are retained. Codex assisted with code, experiments, analysis and writing. Model source: https://github.com/pwangcs/ProxUnroll.`,
'Y = H X W^T + E is a simulation from an existing grayscale image. A detector obtains one integrated response per pattern. Multiple patterns constrain the inverse problem. Reference: Duarte et al., doi:10.1109/MSP.2007.914730.',
'Adjoint, DCT-FISTA, official HQS and official ADMM share the same measurements. The HQS fine-tuned variant also uses the same fixed MAT operator. DCT regularization comes from the original disjoint validation study. References: Beck and Teboulle, doi:10.1137/080716542; Wang et al., arXiv:2505.23180.',
'This architecture comes from the released ProxUnroll work. Stage memory joins adjacent optimization stages within a single image. It does not join video frames. Source image and architecture: https://github.com/pwangcs/ProxUnroll.'
];
for(let i=0;i<4;i++){presentation.slides.items[i].speakerNotes.textFrame.setText(`Planned speaking time: ${times[i]} seconds.\n\n${introNotes[i]}`);guide.push(`## Slide ${i+1}`,introNotes[i]);}
let s=reset(5,'External evaluation protocol','24 Kodak images and three zero-based spaced frames from each of five HEVC B and three HEVC E sequences. The rule is 0, floor(N/3), floor(2N/3), using actual stored lengths. BQTerrace has 601 frames and BasketballDrive 501. We use original raw Y planes, maximal center-square cropping and area resize. Raw Y codes are divided by 255 without range expansion. All are public or laboratory-held data. Sources and SHA-256 are in data/test.json.');
text(s,'24 Kodak images\n24 raw frames from 8 HEVC videos\nCenter-square crop, then 256 × 256 grayscale\n5 clean rates and 2 noise levels with 3 seeds',65,160,1120,210,30);
text(s,'Equal video weights within each HEVC class',65,550,1120,70,29,true,teal);
for(const [i,name] of ['kodim01','hevc_ParkScene_f0080','hevc_Johnny_f0200'].entries())await image(s,path.join(base,'data',name+'.png'),65+300*i,382,180,150,name);
s=reset(6,'Kodak reconstruction across sampling rates','Clean Kodak means over 24 original images. All solvers use the common MAT operator. Nominal 50% caps at 181 × 181 measurements or 49.9893%. Full CSV and independently recomputed float-array metrics: baseline/metrics.csv and baseline/verification.json.');
chart(s,'line',['1%','4%','10%','25%','50%'],Object.keys(names).map(m=>({name:names[m],values:[.01,.04,.1,.25,.5].map(r=>row('Kodak',m,r).psnr),fill:colors[m],line:{fill:colors[m],width:3},marker:{symbol:'circle',size:7}})),{xTitle:'Nominal sampling ratio'});
s=reset(7,'Results by dataset at 10% sampling','Kodak treats each original image as one unit. HEVC B and E first average three frames within each video, then average videos equally. There are five B and three E independent videos. All video results remain in baseline/video_summary.csv. Neural timings use CUDA and DCT-FISTA uses CPU, so timing is not a hardware-matched comparison.');
chart(s,'bar',['Kodak24','HEVC B','HEVC E'],Object.keys(names).map(m=>({name:names[m],values:datasets.map(d=>row(d,m).psnr),fill:colors[m]})));
text(s,'HEVC B: 5 videos       HEVC E: 3 videos',65,608,1100,45,25,false,muted);
s=reset(8,'Bounded HQS fine-tuning','128 seeded DIV2K training originals and 16 disjoint DIV2K validation originals. Kodak and whole HEVC videos are untouched by this new training and selection. Original pretraining exposure is not fully audited. All six sensing factors are frozen. Training uses the upstream proximal trajectory weighted RMSE and a fixed 5e-6 Adam learning rate. Full optimizer, scheduler, RNG and step state are saved separately from selected best weights. DIV2K source: https://data.vision.ee.ethz.ch/cvl/DIV2K/.');
text(s,'128 DIV2K training originals\n16 separate validation originals\nKodak and HEVC reserved for test',65,170,1100,155,32);
text(s,'All sensing factors frozen\nSix-stage reconstruction parameters updated',65,360,1100,125,32,true,teal);
text(s,`${training.completed_steps} total steps    Continued run: ${(training.wall_seconds/60).toFixed(1)} min\nPeak allocated memory: ${(training.peak_memory_mib/1024).toFixed(2)} GiB`,65,536,1100,90,28);
s=reset(9,'Validation selects the checkpoint',`Validation averages 16 originals at clean 10% and 25% rates. Model selection never reads test images. Best step: ${training.best_step}. Bounded training stops at the recorded step/time limit. Full trace: finetune/validation.csv. The short run measured ${smoke.mean_step_seconds.toFixed(2)} seconds per step. Resume verification is recorded separately.`);
chart(s,'line',curve.steps.map(String),[{name:'Validation PSNR',values:curve.psnr,fill:teal,line:{fill:teal,width:3},marker:{symbol:'circle',size:7}}],{xTitle:'Optimization steps',yTitle:'Mean validation PSNR (dB)'});
text(s,`Selected step: ${training.best_step}`,65,613,1100,42,27,true,teal);
s=reset(10,'Test change after HQS fine-tuning','Selected checkpoint minus original HQS under shared measurements. Both 10% and 25% clean rates are reported. Noise results, last-checkpoint results and every image delta are preserved in comparison/paired_comparison.json. Bootstrap uses original Kodak images or whole HEVC videos. A single training seed and few videos limit inference. Positive validation change alone is insufficient evidence of test gain.');
chart(s,'bar',['Kodak24','HEVC B','HEVC E'],[.1,.25].map((r,i)=>({name:`${r*100}% sampling`,values:datasets.map(d=>delta(d,r).delta_psnr),fill:i? '#D78924':teal})),{yTitle:'Fine-tuned minus original PSNR (dB)'});
text(s,'Strong-noise HEVC B/E change: -0.03 / -0.04 dB',65,608,1100,45,25,false,muted);
const bad=comparison.image_deltas.filter(r=>r.method==='hqs_finetuned'&&Number(r.seed)===2026).sort((a,b)=>a.delta_psnr-b.delta_psnr)[0];
s=reset(11,'A difficult held-out test case',`Outcome-based diagnostic only. This image has the smallest selected-model minus original change among the preselected first-seed conditions across all test images. It was not used for selecting the checkpoint or changing the training recipe. Image: ${bad.image}. Sampling: ${Number(bad.cr)*100}%. Noise RMS: ${bad.sigma}. Delta PSNR: ${bad.delta_psnr.toFixed(4)} dB. Every positive and negative delta is preserved. Figure source: retained raw test reconstructions.`);
const imageFile=path.join(base,'data',bad.image+'.png');
const rec=(m)=>path.join(base,'comparison/reconstructions',`${bad.image}_cr${Number(bad.cr)}_noise${Number(bad.sigma)}_seed2026_${m}.png`);
for(const [i,[name,file]] of [['Reference',imageFile],['Original HQS',rec('hqs')],['Fine-tuned HQS',rec('hqs_finetuned')]].entries()){text(s,name,65+i*390,163,365,45,27,true);await image(s,file,65+i*390,229,340,340,name);}
text(s,`${bad.image}    ${Number(bad.cr)*100}% sampling    Noise ${bad.sigma}    Change ${bad.delta_psnr.toFixed(3)} dB`,65,608,1110,44,25,false,muted);
s=reset(12,'Conclusions and limits','We completed remote sync, larger paired evaluation and bounded fixed-operator HQS fine-tuning. Test changes are conditional on this crop/resizing rule and common operator. Original pretraining exposure is not fully audited. One training seed does not establish a general fine-tuning benefit. HEVC frames are independent images, with no previous/future-frame input. No optical camera, full-color system or course-platform submission is claimed. Codex assisted with code, execution, analysis, writing and presentation. Architecture and weights: Wang et al., CVPR 2025, https://github.com/pwangcs/ProxUnroll.');
text(s,'Clean-test PSNR gains: 0.14-0.37 dB at 10% sampling\nStrong-noise HEVC PSNR decreases slightly\nRaw results include failures and recovery checkpoints',65,180,1110,180,31,true,teal);
text(s,'Limits: one training seed and a small DIV2K subset\nCenter cropping changes the benchmark task\nHEVC frames use single-image reconstruction',65,422,1110,150,27);
text(s,'AI disclosure: Codex assisted with implementation, experiments, analysis and writing\nArchitecture and official weights: Wang et al., CVPR 2025',65,613,1110,48,19,false,muted);
const candidate=path.join(tmp,'candidate.pptx');await(await PresentationFile.exportPptx(presentation)).save(candidate);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:output,pythonExecutable:path.join(runtime,'python/bin/python3'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:12,requiredNativeChartOwnerSlides:[6,7,9,10],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:[family],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
const finalDeck=await PresentationFile.importPptx(await FileBlob.load(output));
for(let i=0;i<finalDeck.slides.items.length;i++){const blob=await finalDeck.export({slide:finalDeck.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));}
await fs.writeFile(path.join(root,'coursework/PRESENTATION_GUIDE_LAB220.md'),guide.join('\n\n')+'\n');
console.log(JSON.stringify({output,previews:tmp}));
