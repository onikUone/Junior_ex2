import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class Main {

	//ファイル読み込みメソッド
	public static double[][] readFile(String path) throws IOException{
		List<String[]> list = new ArrayList<String[]>();
		BufferedReader in = new BufferedReader(new FileReader(path));
		String line;
		while((line = in.readLine()) != null){
			list.add(line.split(" "));
		}
		in.close();
		double[][] x = new double[list.size()][list.get(0).length];
		for(int i=0; i<list.size(); i++){
			for(int j=0; j<x[i].length; j++) {
				x[i][j] = Double.parseDouble(list.get(i)[j]);
			}
		}
		return x;
	}

	//ファイル書き込みメソッド
	public static void writeFile(String path, double x[][], double y[][]) throws IOException{
		PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(path)));
		for(int i=0; i<x.length; i++) {
			for(int j=0; j<x[i].length; j++) {
				out.write(String.valueOf(x[i][j]));
				out.write("\t");
			}
			for(int j=0; j<y[i].length; j++) {
				out.write(String.valueOf(y[i][j]));
				out.write("\t");
			}
			out.println("");
		}
		out.close();
	}

	//順方向計算メソッド
	public static OutputNeuron[] forward(InputNeuron input[], InterNeuron inter[], OutputNeuron out[]){
		for(int i=0; i<inter.length; i++){
			inter[i].forward_function(input);
		}
		for(int i=0; i<out.length; i++){
			out[i].forward_function(inter);
		}
		return out;
	}

	public static void main(String[] args) throws IOException {
		//初期パラメータ
		int trainCount = 1;		//学習回数
		int inputNumber = 2;	//入力層個数
		int interNumber = 20;	//中間層個数
		int outputNumber = 3;	//出力層個数
		double preWeight = 0.5;	//結合強度初期値
		double preThreshoud = 0.5; //しきい値初期値
		double preEta = 0.5;	//学習係数初期値
		double preAlpha = 0.8;	//慢性項係数初期値

		//ファイル読み込みPath
//		String readPath = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/inputData.dat";	//Mac(ノートPC)環境
		String readPath = "C:\\Users\\Yuichi Omozaki\\IDrive\\Junior_ex2\\eclipse_workspace\\Junior_ex2\\src\\kadai2.dat";	//Windows(研究室環境)
		//ファイル書き込みPath
//		String writePath = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/outputData.dat";	//Mac(ノートPC)環境
		String writePath = "C:\\Users\\Yuichi Omozaki\\IDrive\\Junior_ex2\\eclipse_workspace\\Junior_ex2\\src\\outputData.dat";	//Windows(研究室環境)

		//ファイル読み込み
		double[][] inputFile;	//datファイル ２次元配列化
		inputFile = readFile(readPath);


		//教師データ作成
		double[][] x = new double[inputFile.length][inputNumber];	//入力ベクトル_教師データ
		double[][] y = new double[inputFile.length][outputNumber];	//出力ベクトル_教師データ
		for(int i=0; i<inputFile.length; i++){
			for(int j=0; j<inputNumber; j++) {
				x[i][j] = inputFile[i][j];
			}
			for(int j=0; j<outputNumber; j++) {
				y[i][j] = inputFile[i][j+inputNumber];
			}
		}
		System.out.println("----------");
		System.out.println("Train Data");
		for(int i=0; i<x.length; i++) {
			System.out.print("x[" + i + "]: ");
			for(int j=0; j<x[i].length; j++) {
				System.out.print(x[i][j] + " ");
			}
			System.out.print("\n");
			System.out.print("y[" + i + "]: ");
			for(int j=0; j<y[i].length; j++) {
				System.out.print(y[i][j] + " ");
			}
			System.out.print("\n\n");
		}
		System.out.println("----------");

		//ニューロン インスタンス作成
		InputNeuron input[] = new InputNeuron[inputNumber];
		for(int i=0; i<input.length; i++) {
			input[i] = new InputNeuron();
		}
		InterNeuron inter[] = new InterNeuron[interNumber];
		for(int i=0; i<inter.length; i++){
			inter[i] = new InterNeuron(inputNumber, preWeight, preThreshoud, preEta, preAlpha);	//コンストラクタには前層の個数を指定 = weightの個数を決定する
		}
		OutputNeuron out[] = new OutputNeuron[outputNumber];
		for(int i=0; i<out.length; i++) {
			out[i] = new OutputNeuron(interNumber, preWeight, preThreshoud, preEta, preAlpha);
		}

		//学習フェーズ
		for(int i=0; i<trainCount; i++){
			//入力層入力
			for(int j=0; j<x.length; j++) {
				for(int k=0; k<input.length; k++) {
					input[k].input(x[j][k]);
				}
			}

			//順方向計算
			forward(input, inter, out);

			//バックプロパゲーション
			for(int j=0; j<y.length; j++) {	//教師データの個数で一周
				//中間層重み更新
				for(int k=0; k<inter.length; k++) {
					inter[k].reWeight(input, out, y[j], k);
				}

				//出力層重み更新
				for(int k=0; k<out.length; k++) {
					out[k].reWeight(inter, y[j][k]);
				}
			}
		}
		System.out.println("Training is Finished.");

		//評価関数
		double e = 0.0;
		for(int i=0; i<x.length; i++) {
			//入力層へ入力
			for(int j=0; j<input.length; j++) {
				input[j].input(x[i][j]);
			}
			for(int j=0; j<outputNumber; j++) {
				e += (y[i][j] - forward(input, inter, out)[j].output()) * (y[i][j] - forward(input, inter, out)[j].output()) / 2;
			}
		}
//		for(int i=0; i< x.length; i++){
//			for(int j=0; j<outputNumber; j++) {
//				e += (y[i][j] - forward(input, inter, out)[j].output()) * (y[i][j] - forward(input, inter, out)[j].output()) / 2;
//			}
//		}
		System.out.println("Count: " + trainCount);
		System.out.println("Error: " + e);

		//学習関数出力
		int h = 100;	//テストデータの刻み幅
		double[][] test_X = new double[h*h][inputNumber];
		double[][] test_Y = new double[h*h][outputNumber];

		for(int i=0; i<h; i++) {
			for(int j=0; j<h; j++) {
				test_X[i*h+j][0] = (double)i/h;
				test_X[i*h+j][1] = (double)j/h;
			}
		}

		//テストデータを入力層へ入力
		for(int i=0; i<test_X.length; i++) {
			for(int j=0; j<input.length; j++) {
				input[j].input(test_X[i][j]);
			}
		}
		for(int i=0; i<test_X.length; i++) {
			for(int j=0; j<outputNumber; j++) {
				test_Y[i][j] = forward(input, inter, out)[j].output();

			}
		}

		//ファイル書き出し
		writeFile(writePath, test_X, test_Y);
	}
}
