import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

	//ファイル読み込みメソッド
	public static double[][] readFile(String path) throws IOException{
		List<String[]> list = new ArrayList<String[]>();
		BufferedReader in = new BufferedReader(new FileReader(path));
		String line;
		while((line = in.readLine()) != null){
			list.add(line.split("\t"));
		}
		in.close();
		double[][] x_n2 = new double[list.size()][list.get(0).length];
		for(int i=0; i<list.size(); i++){
			x_n2[i][0] = Double.parseDouble(list.get(i)[0]);
			x_n2[i][1] = Double.parseDouble(list.get(i)[1]);
		}
		return x_n2;
	}

	//ファイル書き込みメソッド
	public static void writeFile(String path, double x[], double y[]) throws IOException{
		PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(path)));
		for(int i=0; i<x.length; i++){
			out.write(String.valueOf(x[i]));
			out.write("\t");
			out.write(String.valueOf(y[i]));
			out.write("\n");
		}
		out.close();
	}

	//順方向計算メソッド
	public static double forward(double x, InterNeuron inter[], OutputNeuron out, int flg){	//flg 出力層の出力表示フラグ
		for(int i=0; i<inter.length; i++){
			inter[i].clearNet();
		}
		for(int i=0; i<inter.length; i++){
			inter[i].setNet(x, 0);
		}
		for(int i=0; i<inter.length; i++){
			inter[i].calcNet();
		}
		out.clearNet();
		for(int i=0; i<inter.length; i++){
			out.setNet(inter[i].output(), i);
		}
		out.calcNet();
		if(flg == 1){
			System.out.println(out.output());
		}
		return out.output();
	}

	public static void train(double x[], double y[], InterNeuron inter[], OutputNeuron out){
		for(int i=0; i<x.length; i++){	//学習一周につき、入力ベクトル全部計算するためのループ
			//順方向計算
			forward(x[i], inter, out, 0);

			//バックプロパゲーション
			//中間層の重み更新
			for(int j=0; j<inter.length; j++){
				inter[j].reWeight(x[i], y[i], inter[j].output(), out, j);
			}
			//出力層の重み更新
			out.reWeight(y[i], out.output(), inter);
		}
	}

	public static void main(String[] args) throws IOException {
		//初期パラメータ
		int trainCount = 30000;		//学習回数
		int inputNumber = 1;	//入力層個数
		int interNumber = 20;	//中間層個数
//		double preWeight = Math.random();	//結合強度初期値
//		double preThreshoud = Math.random(); //しきい値初期値
		double preEta = 0.5;	//学習係数初期値
		double preAlpha = 0.9;	//慢性項係数初期値

		//ファイル読み込みPath
//		String readPath = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/inputData.dat";	//Mac(ノートPC)環境
		String readPath = "C:\\Users\\Yuichi Omozaki\\IDrive\\Junior_ex1\\eclipse_workspace\\eclipse_ex1\\src\\eclipse_ex1/inputData.dat";	//Windows(研究室環境)
		//ファイル書き込みPath
//		String writePath = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/outputData.dat";	//Mac(ノートPC)環境
		String writePath = "C:\\Users\\Yuichi Omozaki\\IDrive\\Junior_ex1\\eclipse_workspace\\eclipse_ex1\\src\\eclipse_ex1/outputData.dat";	//Windows(研究室環境)

		//ファイル読み込み
		double[][] inputFile;	//datファイル ２次元配列化
		inputFile = readFile(readPath);

		//教師データ作成
		double[] x = new double[inputFile.length];	//入力ベクトル_教師データ
		double[] y = new double[inputFile.length];	//出力ベクトル_教師データ
		for(int i=0; i<inputFile.length; i++){
			x[i] = inputFile[i][0];
			y[i] = inputFile[i][1];
		}
		System.out.println("----------");
		System.out.println("Train Data");
		System.out.println(" x   y ");
		for(int i=0; i<x.length; i++){
			System.out.println(x[i] + " " + y[i]);
		}
		System.out.println("----------");

		//ニューロン インスタンス作成
		InterNeuron inter[] = new InterNeuron[interNumber];
		for(int i=0; i<inter.length; i++){
			inter[i] = new InterNeuron(inputNumber, Math.random(), Math.random(), preEta, preAlpha);	//コンストラクタには前層の個数を指定 = weightの個数を決定する
		}
		OutputNeuron out = new OutputNeuron(interNumber, Math.random(), Math.random(), preEta, preAlpha);

		//学習フェーズ
		for(int i=0; i<trainCount; i++){
			train(x, y, inter, out);
		}
		System.out.println("Training is Finished.");

		//学習関数出力
		double[] test_X = new double[100+1];
		double[] test_Y = new double[100+1];
		Arrays.fill(test_X, 0.0);
		Arrays.fill(test_Y, 0.0);
		test_Y[0] = forward(test_X[0], inter, out, 0);
		for(int i=1; i<test_X.length; i++){
			test_X[i] = test_X[i-1] + 0.01;
			test_Y[i] = forward(test_X[i], inter, out, 0);
		}

		//ファイル書き出し
		writeFile(writePath, test_X, test_Y);

		//評価関数
		double e = 0;
		for(int i=0; i< x.length; i++){
			e += (y[i] - forward(x[i],inter,out,0)) * (y[i] - forward(x[i],inter,out,0)) / 2;
		}
		System.out.println("Count: " + trainCount);
		System.out.println("Error: " + e);
	}
}
