---
title: 余炫｜第15节 MLP学习参数的实现和可视化（交互式）
date: 2026-11-22
---

## 交互式MLP训练可视化
> 可以调整参数，点击开始训练，实时观察MLP权重学习、损失下降、拟合效果
<iframe width="100%" height="620" srcdoc="
<!DOCTYPE html>
<html lang='zh-CN'>
<head>
<meta charset='UTF-8'>
<title>MLP参数可视化</title>
<style>
*{box-sizing:border-box;font-family:system-ui}
body{padding:12px;margin:0}
.ctrl{margin-bottom:10px}
label{margin-right:10px}
button{padding:6px 12px;margin:0 4px}
canvas{border:1px solid #ccc;margin:4px}
</style>
</head>
<body>
<div class='ctrl'>
<label>学习率：<input type='range' id='lr' min='0.001' max='0.2' step='0.001' value='0.05'> <span id='lrVal'>0.05</span></label>
<label>隐藏神经元：<input type='range' id='h' min='2' max='24' step='1' value='8'> <span id='hVal'>8</span></label>
<button id='start'>开始训练</button>
<button id='reset'>重置</button>
</div>
<div>
<canvas id='lossCanvas' width='400' height='220'></canvas>
<canvas id='fitCanvas' width='400' height='220'></canvas>
</div>
<script>
// MLP 类
class MLP {
  constructor(inDim, hDim, outDim) {
    this.w1 = Array.from({length:hDim}, ()=>[Math.random()*2-1])
    this.b1 = Array(hDim).fill(0)
    this.w2 = Array.from({length:outDim}, ()=>Array(hDim).fill(Math.random()*2-1))
    this.b2 = Array(outDim).fill(0)
  }
  relu(x) { return x>0?x:0 }
  forward(x) {
    const h = this.w1.map((w,i)=>this.relu(w[0]*x + this.b1[i]))
    let out = 0
    for(let i=0;i<h.length;i++) out += this.w2[0][i]*h[i]
    return out + this.b2[0]
  }
  train(x,y,lr) {
    const h = this.w1.map((w,i)=>this.relu(w[0]*x + this.b1[i]))
    const pred = this.forward(x)
    const dLoss = pred - y
    // 更新第二层
    for(let i=0;i<h.length;i++){
      this.w2[0][i] -= lr * dLoss * h[i]
    }
    this.b2[0] -= lr * dLoss
    // 更新第一层
    for(let i=0;i<h.length;i++){
      const dh = dLoss * this.w2[0][i] * (h[i]>0?1:0)
      this.w1[i][0] -= lr * dh * x
      this.b1[i] -= lr * dh
    }
    return (pred-y)**2
  }
}
// 数据
const xs = []; const ys = []
for(let i=0;i<100;i++){
  const x = (i/100)*4 -2
  xs.push(x)
  ys.push(Math.sin(3*x)+(Math.random()-0.5)*0.1)
}
let mlp; let lossList = []
const lossCtx = document.getElementById('lossCanvas').getContext('2d')
const fitCtx = document.getElementById('fitCanvas').getContext('2d')
const lrInput = document.getElementById('lr')
const hInput = document.getElementById('h')
document.getElementById('lrVal').innerText = lrInput.value
document.getElementById('hVal').innerText = hInput.value
lrInput.oninput = ()=>document.getElementById('lrVal').innerText = lrInput.value
hInput.oninput = ()=>document.getElementById('hVal').innerText = hInput.value

function drawLoss(){
  const c = lossCtx; const w = c.canvas.width; const h = c.canvas.height
  c.clearRect(0,0,w,h)
  c.beginPath()
  c.strokeStyle='#2277dd'
  for(let i=0;i<lossList.length;i++){
    const px = i/lossList.length*w
    const py = h - lossList[i]*h*0.8
    if(i===0) c.moveTo(px,py)
    else c.lineTo(px,py)
  }
  c.stroke()
}
function drawFit(){
  const c = fitCtx; const w = c.canvas.width; const h = c.canvas.height
  c.clearRect(0,0,w,h)
  // 原始点
  c.fillStyle='#444444'
  for(let i=0;i<xs.length;i++){
    const px = (xs[i]+2)/4*w
    const py = h - (ys[i]+1)/2*h
    c.beginPath(); c.arc(px,py,3,0,Math.PI*2);c.fill()
  }
  // 拟合线
  c.beginPath(); c.strokeStyle='#ee3333'
  for(let i=0;i<xs.length;i++){
    const px = (xs[i]+2)/4*w
    const py = h - (mlp.forward(xs[i])+1)/2*h
    if(i===0) c.moveTo(px,py)
    else c.lineTo(px,py)
  }
  c.stroke()
}
function reset(){
  const hNum = Number(hInput.value)
  mlp = new MLP(1,hNum,1)
  lossList = []
  drawLoss(); drawFit()
}
reset()
let runId = null
document.getElementById('start').onclick = ()=>{
  if(runId) return
  const lr = Number(lrInput.value)
  function loop(){
    let totalLoss = 0
    for(let i=0;i<xs.length;i++){
      totalLoss += mlp.train(xs[i], ys[i], lr)
    }
    lossList.push(totalLoss/xs.length)
    drawLoss(); drawFit()
    if(lossList.length < 600) runId = requestAnimationFrame(loop)
    else runId=null
  }
  loop()
}
document.getElementById('reset').onclick = reset
</script>
</body>
</html>
">
</iframe>

## 实验说明
1. 滑块调节：学习率、隐藏层神经元数量（MLP可学习参数多少由隐藏神经元决定）
2. 点击【开始训练】：自动反向传播、更新权重W和偏置b
3. 左图：损失曲线，观察参数收敛；右图：MLP非线性拟合效果
4. 【重置】清空模型，重新初始化随机参数

## 原理小结
MLP的可学习参数：各层权重W、偏置b；训练依靠梯度下降，根据损失求梯度迭代更新参数。
