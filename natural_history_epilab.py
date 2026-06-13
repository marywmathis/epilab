"""
EpiLab Interactive — Natural History of Disease
Drop this into app.py in the appropriate section.

Usage inside app.py:
    from natural_history_epilab import render_natural_history
    render_natural_history()

Or paste the HTML string directly into your existing
components.v1.html() call if you prefer inline.
"""

import streamlit as st
import streamlit.components.v1 as components

NATURAL_HISTORY_HTML = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;font-size:14px;color:#1f2937;background:#f9fafb}
.tabs{display:flex;gap:8px;padding:0 0 14px;flex-wrap:wrap}
.tab{padding:7px 16px;border-radius:6px;border:1px solid #d1d5db;background:#fff;color:#6b7280;cursor:pointer;font-size:13px;transition:all .15s}
.tab:hover{border-color:#9ca3af;color:#111827}
.tab.on{background:#f3f4f6;border-color:#6b7280;color:#111827;font-weight:500}
.wrap{display:grid;grid-template-columns:210px 1fr 178px;gap:12px;align-items:start}
.panel{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px}
.panel-title{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#9ca3af;margin-bottom:12px}
.rf-item{display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6}
.rf-item:last-of-type{border-bottom:none}
.rf-label{font-size:13px;color:#374151}
.rf-check{width:15px;height:15px;accent-color:#16a34a;cursor:pointer}
.rf-slider-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #f3f4f6}
.rf-slider-label{font-size:13px;color:#374151;flex:1}
.rf-slider-row input[type=range]{width:72px}
.rf-slider-val{font-size:12px;color:#6b7280;min-width:14px;text-align:right}
.sus-section{margin-top:10px;padding-top:10px;border-top:1px solid #f3f4f6}
.sus-row{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;color:#6b7280}
.sus-val{font-weight:600;color:#111827}
.sus-track{height:5px;background:#f3f4f6;border-radius:3px;overflow:hidden;margin-bottom:5px}
.sus-fill{height:100%;border-radius:3px;transition:width .35s,background .35s}
.sus-active{font-size:11px;color:#9ca3af;line-height:1.5;min-height:28px;margin-bottom:2px}
.reset-btn{display:block;width:100%;margin-top:10px;padding:6px 0;border-radius:6px;border:1px solid #e5e7eb;background:transparent;color:#9ca3af;font-size:12px;cursor:pointer;transition:all .15s;text-align:center}
.reset-btn:hover{border-color:#6b7280;color:#374151;background:#f9fafb}
.c-title{font-size:16px;font-weight:600;color:#111827;margin-bottom:2px}
.c-sub{font-size:12px;color:#9ca3af;margin-bottom:10px}
.legend{display:flex;gap:14px;margin-bottom:12px}
.leg{display:flex;align-items:center;gap:5px;font-size:12px;color:#6b7280}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.stage-card{border-radius:8px;border:1px solid transparent;padding:13px 15px;margin-bottom:4px;transition:all .3s}
.stage-card.dim{background:#f9fafb;border-color:#f3f4f6;opacity:.6}
.stage-card.low{background:#f0fdf4;border-color:#86efac}
.stage-card.mod{background:#fffbeb;border-color:#fcd34d}
.stage-card.high{background:#fff1f2;border-color:#fca5a5}
.sc-name{font-size:14px;font-weight:600;margin-bottom:3px}
.sc-desc{font-size:13px;margin-bottom:3px}
.sc-detail{font-size:12px}
.stage-card.dim .sc-name,.stage-card.dim .sc-desc,.stage-card.dim .sc-detail{color:#d1d5db}
.stage-card.low .sc-name{color:#15803d}.stage-card.low .sc-desc{color:#166534}.stage-card.low .sc-detail{color:#166534;opacity:.8}
.stage-card.mod .sc-name{color:#92400e}.stage-card.mod .sc-desc{color:#78350f}.stage-card.mod .sc-detail{color:#78350f;opacity:.8}
.stage-card.high .sc-name{color:#9f1239}.stage-card.high .sc-desc{color:#881337}.stage-card.high .sc-detail{color:#881337;opacity:.8}
.arrow{text-align:center;color:#d1d5db;font-size:16px;line-height:1.2;margin:1px 0}
.sys-item{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:13px;color:#374151;border-bottom:1px solid #f3f4f6}
.sys-item:last-child{border-bottom:none}
.outcome-item{display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f3f4f6}
.outcome-item:last-child{border-bottom:none}
.out-name{font-size:12px;color:#374151}
.out-badge{font-size:10px;font-weight:600;padding:2px 7px;border-radius:10px;transition:all .3s}
.badge-unlikely{background:#f0fdf4;color:#15803d;border:1px solid #86efac}
.badge-possible{background:#fffbeb;color:#92400e;border:1px solid #fcd34d}
.badge-likely{background:#fff1f2;color:#9f1239;border:1px solid #fca5a5}
@media(max-width:600px){.wrap{grid-template-columns:1fr}}
</style>

<div class="tabs">
  <button class="tab on" onclick="sw('htn',this)">Hypertension</button>
  <button class="tab" onclick="sw('covid',this)">COVID-19</button>
  <button class="tab" onclick="sw('t2dm',this)">Type 2 Diabetes</button>
  <button class="tab" onclick="sw('cad',this)">Coronary Artery Disease</button>
  <button class="tab" onclick="sw('cirr',this)">Liver Cirrhosis</button>
  <button class="tab" onclick="sw('cerv',this)">Cervical Cancer</button>
</div>

<div class="wrap">
  <div class="panel">
    <div class="panel-title">Risk factors</div>
    <div id="rf-list"></div>
    <div class="sus-section">
      <div class="sus-row">
        <span>Susceptibility score</span>
        <span><span class="sus-val" id="sus-num">0%</span>&nbsp;<span id="sus-level">· Low</span></span>
      </div>
      <div class="sus-track"><div class="sus-fill" id="sus-fill" style="width:0%;background:#16a34a"></div></div>
      <div class="sus-active" id="sus-active">No active risk factors.</div>
    </div>
    <button class="reset-btn" onclick="resetCurrent()">&#8635; Reset risk factors</button>
  </div>

  <div class="panel" style="display:flex;flex-direction:column;gap:0">
    <div class="c-title" id="c-title"></div>
    <div class="c-sub">Hover any stage for clinical detail</div>
    <div class="legend">
      <div class="leg"><div class="dot" style="background:#16a34a"></div>Low risk</div>
      <div class="leg"><div class="dot" style="background:#d97706"></div>Moderate</div>
      <div class="leg"><div class="dot" style="background:#e11d48"></div>High risk</div>
    </div>
    <div id="stages-list"></div>
  </div>

  <div style="display:flex;flex-direction:column;gap:12px">
    <div class="panel">
      <div class="panel-title">Systems affected</div>
      <div id="sys-list"></div>
    </div>
    <div class="panel">
      <div class="panel-title">Outcomes</div>
      <div id="out-list"></div>
    </div>
  </div>
</div>

<script>
const D={
  htn:{name:'Hypertension',risks:[
    {id:'age',label:'Age > 55',w:15},{id:'obese',label:'Obesity (BMI > 30)',w:20},
    {id:'smoking',label:'Current smoker',w:20},{id:'sodium',label:'High sodium diet',w:15},
    {id:'sedentary',label:'Sedentary lifestyle',w:15},{id:'family',label:'Family history',w:15},
    {id:'stress',label:'Chronic stress',w:10},{id:'alcohol',label:'Alcohol use (heavy)',w:10}],
  stages:[
    {name:'Susceptibility',desc:'Genetic + lifestyle factors create vulnerability',detail:'No symptoms. BP normal. Risk factors accumulate silently.',level:'low',thresh:0},
    {name:'Subclinical disease',desc:'BP rising; early vascular changes, no symptoms',detail:'BP 130–139/80–89 mmHg. Silent LV remodeling. Microalbuminuria may emerge.',level:'low',thresh:20},
    {name:'Clinical disease',desc:'Overt hypertension; organ stress begins',detail:'BP \u2265 140/90 mmHg. Headache, epistaxis possible. Retinal changes. Antihypertensives indicated.',level:'mod',thresh:40},
    {name:'Disability',desc:'End-organ damage \u2014 heart, kidneys, brain',detail:'HFpEF, CKD, hypertensive retinopathy. Quality of life declines. Multidisciplinary care.',level:'high',thresh:65},
    {name:'Death / outcomes',desc:'MI, stroke, ESRD, aortic dissection',detail:'Preventable at every prior stage with treatment adherence.',level:'high',thresh:85}],
  systems:[{name:'Cardiovascular',thresh:20},{name:'Renal',thresh:40},{name:'Neurological',thresh:50},{name:'Ophthalmological',thresh:40},{name:'Peripheral vascular',thresh:60}],
  outcomes:[{name:'Stroke',thresh:[0,40,70]},{name:'Myocardial infarction',thresh:[0,40,70]},{name:'Heart failure',thresh:[0,50,75]},{name:'Renal failure (ESRD)',thresh:[0,55,80]},{name:'Premature death',thresh:[0,65,85]}]},

  covid:{name:'COVID-19',risks:[
    {id:'age65',label:'Age > 65',w:20},{id:'unvacc',label:'Unvaccinated',w:25},
    {id:'obese2',label:'Obesity (BMI > 30)',w:15},{id:'dm',label:'Diabetes',w:15},
    {id:'immuno',label:'Immunocompromised',w:20},{id:'cardio',label:'Cardiovascular disease',w:15},
    {id:'lung',label:'Chronic lung disease',w:15},
    {id:'exposure',label:'Exposure intensity',w:0,slider:true,min:1,max:5,val:2},
    {id:'mask',label:'No masking / PPE',w:10},{id:'crowd',label:'Crowded living / work',w:10}],
  stages:[
    {name:'Exposure & susceptibility',desc:'Contact with SARS-CoV-2; host defense determines infection',detail:'Incubation: 2\u201314 days (mean ~5 days)',level:'low',thresh:0},
    {name:'Incubation / viral replication',desc:'Replication in upper respiratory tract; often asymptomatic',detail:'Duration: 2\u20135 days \u2014 many remain asymptomatic',level:'low',thresh:10},
    {name:'Mild to moderate illness',desc:'Fever, cough, fatigue, anosmia; immune response active',detail:'Most patients recover here (7\u201314 days)',level:'mod',thresh:30},
    {name:'Severe / critical illness',desc:'Pneumonia, hypoxia, cytokine storm, ARDS',detail:'Risk: low \u2014 especially with vaccination',level:'high',thresh:55}],
  systems:[{name:'Respiratory',thresh:10},{name:'Cardiovascular',thresh:30},{name:'Neurological',thresh:30},{name:'Renal',thresh:50},{name:'Coagulation',thresh:40}],
  outcomes:[{name:'Long COVID',thresh:[0,25,60]},{name:'ARDS / resp failure',thresh:[0,45,65]},{name:'VTE / thromboembolism',thresh:[0,50,70]},{name:'Acute kidney injury',thresh:[0,55,75]},{name:'Death',thresh:[0,60,80]}]},

  t2dm:{name:'Type 2 Diabetes',risks:[
    {id:'bmi',label:'Obesity (BMI > 30)',w:25},{id:'ifg',label:'Impaired fasting glucose',w:25},
    {id:'fam',label:'Family history DM',w:20},{id:'age2',label:'Age > 45',w:15},
    {id:'sed2',label:'Physical inactivity',w:15},{id:'htn2',label:'Hypertension',w:15},
    {id:'pcos',label:'PCOS / gestational DM hx',w:15},{id:'diet',label:'High glycemic diet',w:10}],
  stages:[
    {name:'Insulin resistance',desc:'Cells less responsive to insulin; compensatory hyperinsulinemia',detail:'FPG 70\u201399 mg/dL. No symptoms. Fully reversible with lifestyle change.',level:'low',thresh:0},
    {name:'Prediabetes',desc:'FPG 100\u2013125 mg/dL; HbA1c 5.7\u20136.4%; beta-cell stress',detail:'Postprandial spikes begin. Vessel damage accumulating silently.',level:'low',thresh:20},
    {name:'Overt type 2 diabetes',desc:'FPG \u2265 126 mg/dL; HbA1c \u2265 6.5%; classic symptoms',detail:'Polyuria, polydipsia, fatigue. Metformin first-line. SGLT-2i if cardiorenal risk.',level:'mod',thresh:40},
    {name:'Microvascular complications',desc:'Retinopathy, nephropathy, neuropathy',detail:'HbA1c control critical. Annual eye and foot exams.',level:'high',thresh:60},
    {name:'Macrovascular / end-stage',desc:'CVD, ESRD, amputation, blindness',detail:'MI and stroke risk \u00d72\u20134. ESRD. Lower-limb amputation.',level:'high',thresh:80}],
  systems:[{name:'Endocrine / metabolic',thresh:0},{name:'Cardiovascular',thresh:30},{name:'Renal',thresh:40},{name:'Ophthalmological',thresh:40},{name:'Peripheral nervous system',thresh:40}],
  outcomes:[{name:'Blindness (retinopathy)',thresh:[0,45,70]},{name:'ESRD / dialysis',thresh:[0,55,80]},{name:'Lower limb amputation',thresh:[0,55,78]},{name:'Myocardial infarction',thresh:[0,45,70]},{name:'Premature death',thresh:[0,60,85]}]},

  cad:{name:'Coronary Artery Disease',risks:[
    {id:'age_m',label:'Age > 45 (male) / > 55 (female)',w:20},{id:'smoking_c',label:'Current smoker',w:25},
    {id:'htn_c',label:'Hypertension',w:20},{id:'hld',label:'Hyperlipidemia (LDL > 130)',w:20},
    {id:'dm_c',label:'Diabetes mellitus',w:20},{id:'fam_c',label:'Family history (1st degree, early onset)',w:20},
    {id:'obese_c',label:'Obesity (BMI > 30)',w:15},{id:'sed_c',label:'Sedentary lifestyle',w:10},
    {id:'ckd_c',label:'Chronic kidney disease',w:10}],
  stages:[
    {name:'Susceptibility',desc:'Risk factor accumulation; endothelial vulnerability',detail:'No symptoms. Lipid profiles may be abnormal. ASCVD risk score elevated. Primary prevention window.',level:'low',thresh:0},
    {name:'Endothelial dysfunction & early atherosclerosis',desc:'Fatty streaks form; LDL oxidizes within intima',detail:'Silent process beginning in adolescence. Coronary calcium score (CAC) may show early calcification. No angina.',level:'low',thresh:20},
    {name:'Stable coronary artery disease',desc:'Atherosclerotic plaques narrow lumen; exertional angina',detail:'Chest pain with exertion, relieved by rest. Stress testing positive. Statins, aspirin, beta-blockers indicated.',level:'mod',thresh:40},
    {name:'Acute coronary syndrome',desc:'Plaque rupture or erosion; thrombus formation',detail:'Unstable angina, NSTEMI, or STEMI. Emergency PCI or fibrinolysis. High mortality without rapid treatment.',level:'high',thresh:65},
    {name:'Heart failure / death',desc:'Myocardial infarction leads to ventricular dysfunction',detail:'Ischemic cardiomyopathy. EF < 40%. ICD, CRT, or transplant consideration.',level:'high',thresh:82}],
  systems:[{name:'Cardiovascular',thresh:0},{name:'Renal',thresh:40},{name:'Neurological (embolic stroke)',thresh:50},{name:'Peripheral vascular',thresh:40},{name:'Metabolic / endocrine',thresh:30}],
  outcomes:[{name:'Unstable angina / NSTEMI',thresh:[0,35,65]},{name:'STEMI',thresh:[0,50,75]},{name:'Ischemic cardiomyopathy',thresh:[0,55,78]},{name:'Sudden cardiac death',thresh:[0,55,80]},{name:'Ischemic stroke',thresh:[0,50,72]}]},

  cirr:{name:'Liver Cirrhosis',risks:[
    {id:'etoh',label:'Alcohol use disorder',w:30},{id:'hcv',label:'Hepatitis C (untreated)',w:30},
    {id:'hbv',label:'Hepatitis B (untreated)',w:25},{id:'nafld',label:'Metabolic dysfunction (MASLD/NAFLD)',w:20},
    {id:'obese_r',label:'Obesity (BMI > 30)',w:15},{id:'dm_r',label:'Diabetes mellitus',w:15},
    {id:'iron',label:'Hemochromatosis / iron overload',w:15},{id:'drug',label:'Hepatotoxic drug exposure',w:10}],
  stages:[
    {name:'Susceptibility',desc:'Hepatic exposure to injurious agents; genetic vulnerability',detail:'Liver function normal. No fibrosis. Alcohol use, viral infection, or metabolic syndrome present. Fully reversible if exposure eliminated.',level:'low',thresh:0},
    {name:'Hepatic steatosis & inflammation',desc:'Fat accumulation and hepatocyte injury; elevated enzymes',detail:'ALT/AST elevated. Liver enlarged on ultrasound. Steatohepatitis (NASH/ASH). Still reversible with behavior change or antiviral treatment.',level:'low',thresh:20},
    {name:'Fibrosis (F1\u2013F3)',desc:'Collagen deposition replaces hepatocytes; architecture distorted',detail:'Fibroscan or biopsy confirms fibrosis staging. Portal hypertension developing. Partial reversibility with treatment.',level:'mod',thresh:38},
    {name:'Compensated cirrhosis (F4)',desc:'Advanced scarring; liver still maintaining function',detail:'Splenomegaly, thrombocytopenia, varices may be present. Child-Pugh A. HCC surveillance begins (ultrasound q6 months).',level:'mod',thresh:55},
    {name:'Decompensated cirrhosis',desc:'Liver can no longer compensate; clinical complications emerge',detail:'Ascites, variceal bleeding, hepatic encephalopathy, SBP. Child-Pugh B/C. MELD score guides transplant listing.',level:'high',thresh:72},
    {name:'End-stage liver disease / HCC',desc:'Hepatocellular carcinoma or liver failure',detail:'MELD > 15: transplant evaluation. HCC develops in 1\u20135% per year in cirrhotic patients. 5-year survival without transplant: < 20%.',level:'high',thresh:85}],
  systems:[{name:'Hepatic / biliary',thresh:0},{name:'Gastrointestinal (varices)',thresh:40},{name:'Renal (hepatorenal syndrome)',thresh:60},{name:'Neurological (encephalopathy)',thresh:60},{name:'Hematologic / coagulation',thresh:40}],
  outcomes:[{name:'Variceal hemorrhage',thresh:[0,45,70]},{name:'Ascites / SBP',thresh:[0,50,72]},{name:'Hepatic encephalopathy',thresh:[0,55,75]},{name:'Hepatocellular carcinoma',thresh:[0,45,68]},{name:'Liver failure / death',thresh:[0,60,82]}]},

  cerv:{name:'Cervical Cancer',risks:[
    {id:'hpv',label:'HPV infection (high-risk strains)',w:40},{id:'unvacc_h',label:'Unvaccinated against HPV',w:25},
    {id:'noscr',label:'No Pap / HPV screening',w:25},{id:'smoking_h',label:'Current smoker',w:15},
    {id:'immuno_h',label:'Immunocompromised (HIV+)',w:20},{id:'multipart',label:'Multiple sexual partners',w:10},
    {id:'earlysx',label:'Early sexual debut (< 16)',w:10},{id:'chlamydia',label:'Co-infection (chlamydia)',w:10}],
  stages:[
    {name:'Susceptibility',desc:'HPV exposure risk; host immune factors determine clearance',detail:'No HPV infection yet. Vaccination before sexual debut provides ~97% protection against HPV 16/18. Primary prevention window is open.',level:'low',thresh:0},
    {name:'HPV infection',desc:'High-risk HPV acquired; most infections clear within 2 years',detail:'80% of sexually active people infected in lifetime. Persistent high-risk HPV (16, 18, 31, 45) drives dysplasia.',level:'low',thresh:25},
    {name:'Cervical dysplasia (CIN 1\u20132)',desc:'Low-to-moderate grade squamous intraepithelial lesion',detail:'Detected on Pap smear / colposcopy. CIN 1: usually regresses. CIN 2: monitor or treat. Still preinvasive \u2014 treatment highly effective.',level:'mod',thresh:45},
    {name:'High-grade dysplasia (CIN 3 / CIS)',desc:'Severe dysplasia or carcinoma in situ; cells confined to epithelium',detail:'LEEP or cone biopsy curative. 5-year risk of progression to invasion ~30% if untreated. Screening programs catch disease here.',level:'mod',thresh:60},
    {name:'Invasive cervical cancer (Stage I\u2013II)',desc:'Malignant cells invade stroma; local disease',detail:'Stage I: confined to cervix. Stage II: upper vagina or parametrium. Surgery + radiation \u00b1 chemo. 5-year survival Stage I: ~90%.',level:'high',thresh:72},
    {name:'Advanced / metastatic (Stage III\u2013IV)',desc:'Regional lymph nodes, bladder, rectum, or distant metastases',detail:'Stage IV: bladder, rectum, or distant mets. Chemoradiation palliative. 5-year survival Stage IV: ~15\u201320%. Entirely preventable with vaccination + screening.',level:'high',thresh:85}],
  systems:[{name:'Reproductive / gynecologic',thresh:0},{name:'Urinary (ureteral obstruction)',thresh:55},{name:'Gastrointestinal (rectal involvement)',thresh:70},{name:'Lymphatic / nodal',thresh:60},{name:'Hematologic (anemia)',thresh:50}],
  outcomes:[{name:'CIN 3 / carcinoma in situ',thresh:[0,40,65]},{name:'Invasive cervical cancer',thresh:[0,55,72]},{name:'Fistula (bladder / rectal)',thresh:[0,65,80]},{name:'Ureteral obstruction / renal failure',thresh:[0,65,80]},{name:'Death',thresh:[0,65,85]}]}
};

let cur='htn',state={};

function score(){
  const d=D[cur];let tot=0,act=0;
  d.risks.forEach(r=>{
    if(r.slider){const v=state[r.id]??r.val;const c=Math.round(((v-r.min)/(r.max-r.min))*25);tot+=25;act+=c;}
    else{tot+=r.w;if(state[r.id])act+=r.w;}
  });
  return tot>0?Math.round(act/tot*100):0;
}
function activeLabels(){
  return D[cur].risks.filter(r=>r.slider?(state[r.id]??r.val)>r.min:state[r.id]).map(r=>r.label);
}
function badgeClass(sc,t){return sc<t[1]?'badge-unlikely':sc<t[2]?'badge-possible':'badge-likely';}
function badgeLabel(sc,t){return sc<t[1]?'Unlikely':sc<t[2]?'Possible':'Likely';}
function stageClass(s,sc){return sc<s.thresh?'dim':s.level;}
function sysColor(thresh,sc){
  if(sc<thresh)return '#d1d5db';
  if(sc<thresh+25)return '#86efac';
  if(sc<thresh+50)return '#fcd34d';
  return '#f87171';
}
function resetCurrent(){state={};render();}
function render(){
  const d=D[cur];const sc=score();
  document.getElementById('sus-num').textContent=sc+'%';
  document.getElementById('sus-level').textContent='\u00b7 '+(sc<35?'Low':sc<65?'Moderate':'High');
  const fill=document.getElementById('sus-fill');
  fill.style.width=sc+'%';
  fill.style.background=sc<35?'#16a34a':sc<65?'#d97706':'#e11d48';
  const al=activeLabels();
  document.getElementById('sus-active').textContent=al.length?'Active: '+al.join(', ')+'.':'No active risk factors.';
  document.getElementById('c-title').textContent='Natural history of '+d.name;
  let rfh='';
  d.risks.forEach(r=>{
    if(r.slider){
      const v=state[r.id]??r.val;
      rfh+=`<div class="rf-slider-row"><span class="rf-slider-label">${r.label}</span><input type="range" min="${r.min}" max="${r.max}" value="${v}" step="1" oninput="state['${r.id}']=+this.value;document.getElementById('sv-${r.id}').textContent=this.value;render()"><span class="rf-slider-val" id="sv-${r.id}">${v}</span></div>`;
    }else{
      rfh+=`<div class="rf-item"><span class="rf-label">${r.label}</span><input class="rf-check" type="checkbox" ${state[r.id]?'checked':''} onchange="state['${r.id}']=this.checked;render()"></div>`;
    }
  });
  document.getElementById('rf-list').innerHTML=rfh;
  let sh='';
  d.stages.forEach((s,i)=>{
    sh+=`<div class="stage-card ${stageClass(s,sc)}" title="${s.detail}"><div class="sc-name">${s.name}</div><div class="sc-desc">${s.desc}</div><div class="sc-detail">${s.detail}</div></div>`;
    if(i<d.stages.length-1)sh+=`<div class="arrow">\u2193</div>`;
  });
  document.getElementById('stages-list').innerHTML=sh;
  document.getElementById('sys-list').innerHTML=d.systems.map(s=>`<div class="sys-item"><div class="dot" style="background:${sysColor(s.thresh,sc)}"></div><span>${s.name}</span></div>`).join('');
  document.getElementById('out-list').innerHTML=d.outcomes.map(o=>`<div class="outcome-item"><span class="out-name">${o.name}</span><span class="out-badge ${badgeClass(sc,o.thresh)}">${badgeLabel(sc,o.thresh)}</span></div>`).join('');
}
function sw(key,btn){
  cur=key;state={};
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  btn.classList.add('on');
  render();
}
render();
</script>
"""


def render_natural_history():
    """
    Render the Natural History of Disease interactive module.
    Call this inside your app.py section for this content area.

    Height is set to 900 to accommodate the six-stage diseases
    (liver cirrhosis, cervical cancer). Adjust down to 820 if
    you remove those and keep only the five-stage diseases.
    """
    st.markdown("### Natural History of Disease")
    st.caption(
        "Adjust risk factors to see how disease progression and outcomes change. "
        "Toggle risk factors on or off — stages illuminate as susceptibility rises."
    )
    components.html(NATURAL_HISTORY_HTML, height=900, scrolling=False)
