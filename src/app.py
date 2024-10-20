# 必要なモジュールのインポート
import torch
from rice_blast import transform, Net # rice_blast.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # 学習済みモデルの重み（rice_blast.pt）を読み込み
    net.load_state_dict(torch.load('./src/rice_blast.pt', map_location=torch.device('cpu')))
    # データの前処理
    img = transform(img)
    img = img.unsqueeze(0) # 一次元増やす
    # 推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

# 推論したラベルから　いもち病　か　否かを返す関数
def getName(label):
    if label==0:
         return 'いもち病です！'
    if label==1:
         return 'いもち病ではありません！'

# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg' ])

# 拡張子が適切かどうかをチェック
def allowed_file(filename):
     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
     # リクエストがポストかどうかの判別
     if request.method == 'POST':
          # ファイルがなかった場合の処理
         if 'filename' not in request.files:
            return redirect(request.url)
         # データの取り出し
         file = request.files['filename']
         # ファイルのチェック
         if file and allowed_file(file.filename):
         
            # 画像ファイルに対する処理
            # 画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            # 画像データをバッファに書き込む
            image.save(buf, 'png')
            # バイナリデータをbase64でエンコードして　utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            # HTML側の src の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            rice_blastName = getName(pred)
            return render_template('result.html', rice_blastName = rice_blastName, image=base64_data)
     
     # GET メソッドの定義
     elif request.method == 'GET':
        return render_template('index.html')
 
 # アプリケーションの実行の定義 
if __name__ == '__main__':
    app.run(debug=True)     