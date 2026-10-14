"""Package only the current reading/presentation materials, with checked hashes."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/materials_final_20261014'
OUTPUT=ROOT/'output/delivery/ProxUnroll_coursework_delivery_20261014.zip'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    audit=json.loads((BASE/'completion_audit.json').read_text())
    assert audit['slides']==12 and audit['technical_pages']==4 and audit['literature_pages']==3
    files={
        'technical_report.pdf':'output/pdf/technical_report_flow_20261014.pdf',
        'literature_review.pdf':'output/pdf/literature_review_final_20261014.pdf',
        'presentation.pptx':'output/presentation/course_presentation_final_20261014_v2.pptx',
        'speech_zh.md':'coursework/讲稿_最终交付_20261014.md',
        'defense_zh.md':'coursework/DEFENSE_FINAL_20261014.md',
        'presentation_notes_en.md':'coursework/PRESENTATION_GUIDE_FINAL_20261014.md',
        'requirements_status_zh.md':'coursework/REQUIREMENTS_STATUS_FINAL_20261014.md',
        'reproduce_zh.md':'coursework/REPRODUCE_FINAL_20261014.md',
        'results_20000_zh.md':'coursework/HQS_20000_RESULTS_20261014.md',
        'image_attribution.md':'runs/nanjing_web30_20261009/ATTRIBUTION.md',
        'evidence/materials_audit.json':'runs/materials_final_20261014/completion_audit.json',
        'evidence/evaluation_audit.json':'runs/hqs_20000_eval_20261014/completion_audit.json',
        'evidence/evaluation_lock.json':'runs/hqs_20000_eval_20261014/lock.json',
        'csv/four_method_dataset_scores.csv':'runs/nanjing_web30_20261009/comparison/dataset_summary.csv',
        'csv/previous_primary_pairs.csv':'runs/hqs_ablation_20261013/evaluation_primary/paired_summary.csv',
    }
    for phase in ['primary','regression']:
        for name in ['all_model_summary.csv','paired_summary.csv','video_mse_summary.csv','metrics.csv','negative_cases.csv']:
            files[f'csv/{phase}_{name}']=f'runs/hqs_20000_eval_20261014/evaluation_{phase}/{name}'
    content={name:(ROOT/path).read_bytes() for name,path in files.items()}
    for name,key in [('technical_report.pdf','technical_sha256'),('literature_review.pdf','literature_sha256'),('presentation.pptx','ppt_sha256')]:
        assert sha(content[name])==audit[key],name
    content['README.md']='''# 单像素成像课程项目：最新材料包（2026-10-14）

此包只保留最新版本，旧报告与旧PPT在项目仓库中继续保留。

- technical_report.pdf：英文技术报告4页，连续双栏，包含20000步评测。
- literature_review.pdf：英文文献报告3页，经典Duarte 2008与Wang 2025比较，实践内容已更新。
- presentation.pptx：英文12页PPT，含可编辑图表/表格、英文备注、来源与AI声明。
- speech_zh.md：按12页逐页讲解的中文讲稿，计划10分钟。
- defense_zh.md：中文答辩准备；requirements_status_zh.md：要求与剩余学生事项。
- reproduce_zh.md：代码、环境、数据/权重位置、核验命令与最小推理说明。
- results_20000_zh.md与csv/：完整结果、每视频统计和负例，不只保留好图。
- image_attribution.md：网络照片作者与许可。照片由外部作者拍摄，团队自行整理。
- evidence/与SHA256SUMS.json：来源、核验与本包内容哈希。

主线完成20000次更新，验证选择19500，终点20000单独保留。相对上一轮11500，主要52图10%采样时MSE增加0.83%（干净）/0.41%（较强噪声），SSIM略升。其他数据集小幅、混合变化。最后8000步没有新增普遍5–10%改善。PPT第8页统一相对10000，末页回归表相对11500，讲稿明确区别。失败图来自11500，不能当19500重建。

代码仓库：https://github.com/congcongcong1/ProxUnroll-main。
仓库当前入口：COURSEWORK_DELIVERY_20261014.md。
完整原图、大权重、测量矩阵、全部浮点重建/PNG和环境保留服务器192.168.1.244的luozc_mlvc容器，项目/workspace/ProxUnroll-main。包中没有Mac环境或大数据，不是完整代码运行环境。

这是已经整理的材料包，没有代交课程平台。学生还需核对姓名学号、理解代码、在自己的电脑打开PPT、计时排练并自行提交。课堂平台如只允许指定格式，请按其实际规则选择文件，不需上传问答或内部审计。文献阅读如另行汇报，需要按课堂安排准备。
'''.encode()
    sums={name:sha(data) for name,data in content.items()}
    content['SHA256SUMS.json']=(json.dumps(sums,ensure_ascii=False,indent=2)+'\n').encode()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    assert not OUTPUT.exists(),'Preserve existing delivery archive; select a new dated output'
    with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=8) as archive:
        for name,data in content.items():archive.writestr(name,data)
    with zipfile.ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())==set(content)
        for name,data in content.items():assert archive.read(name)==data,name
        for name,digest in json.loads(archive.read('SHA256SUMS.json')).items():assert sha(archive.read(name))==digest,name
    result={'archive':str(OUTPUT.relative_to(ROOT)),'sha256':sha(OUTPUT.read_bytes()),'bytes':OUTPUT.stat().st_size,
            'members':len(content),'all_members_reopened_and_sha256_verified':True,'repository_sources':files,
            'scope':'Current coursework artifacts, selected CSV and provenance; no datasets, large weights, local environments or platform submission'}
    (BASE/'package_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
