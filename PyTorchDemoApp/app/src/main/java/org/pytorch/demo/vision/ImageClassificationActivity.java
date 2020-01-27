package org.pytorch.demo.vision;

import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import static java.util.Collections.*;
import static java.util.stream.Collectors.toList;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_WIDTH = 128;
  private static final int INPUT_TENSOR_HEIGHT = 32;
  private static final int TOP_K = 3;
  private static final int MOVING_AVG_PERIOD = 10;
  private static final String FORMAT_MS = "%dms";
  private static final String FORMAT_AVG_MS = "avg:%.0fms";

  private static final String FORMAT_FPS = "%.1fFPS";
  public static final String SCORES_FORMAT = "%.2f";

  private static final String BLANK = " ";
  private static final String ALPHABET = BLANK + "0123456789abcdefghijklmnopqrstuvwxyzøæå";
  private static final int PRED_LENGTH = INPUT_TENSOR_WIDTH / 4;
  private static final double MIN_CONF = 0.0;  // Needs special care with topK

  static class AnalysisResult {

    private final String[] topNClassNames;
    private final float[] topNScores;
    private final long analysisDuration;
    private final long moduleForwardDuration;

    public AnalysisResult(String[] topNClassNames, float[] topNScores,
                          long moduleForwardDuration, long analysisDuration) {
      this.topNClassNames = topNClassNames;
      this.topNScores = topNScores;
      this.moduleForwardDuration = moduleForwardDuration;
      this.analysisDuration = analysisDuration;
    }
  }

  private boolean mAnalyzeImageErrorState;
  private ResultRowView[] mResultRowViews = new ResultRowView[TOP_K];
  private TextView mFpsText;
  private TextView mMsText;
  private TextView mMsAvgText;
  private Module mModule;
  private String mModuleAssetName;
  private FloatBuffer mInputTensorBuffer;
  private Tensor mInputTensor;
  private long mMovingAvgSum = 0;
  private Queue<Long> mMovingAvgQueue = new LinkedList<>();

  static class PredConf {
    public final String pred;
    public final Float conf;

    public PredConf(String pred, Float conf) {
      this.pred = pred;
      this.conf = conf;
    }
  }

  static class Stuff {
    public final List<Integer> rawPred;
    public final List<Integer> newPred;
    public final List<List<Integer>> sortedIdxs;
    public final Float conf;
    public final Float newConf;
    public final List<Float> rawConf;
    public final List<Float> newRawConf;

    public Stuff(List<Integer> rawPred, List<Integer> newPred, List<List<Integer>> sortedIdxs,
                 Float conf, Float newConf, List<Float> rawConf, List<Float> newRawConf) {
      this.rawPred = rawPred;
      this.newPred = newPred;
      this.sortedIdxs = sortedIdxs;
      this.conf = conf;
      this.newConf = newConf;
      this.rawConf = rawConf;
      this.newRawConf = newRawConf;
    }
  }

  static class ConfIdx {
    public final Float conf;
    public final int idx;

    public ConfIdx(Float conf, int idx) {
      this.conf = conf;
      this.idx = idx;
    }
  }

  static class SortByConf implements Comparator<ConfIdx> {
    @Override
    public int compare(ConfIdx a, ConfIdx b)
    {
      return a.conf.compareTo(b.conf);
    }
  }

  @Override
  protected int getContentViewLayoutId() {
    return R.layout.activity_image_classification;
  }

  @Override
  protected TextureView getCameraPreviewTextureView() {
    return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
        .inflate()
        .findViewById(R.id.image_classification_texture_view);
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    final ResultRowView headerResultRowView =
        findViewById(R.id.image_classification_result_header_row);
    headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
    headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

    mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
    mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
    mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);

    mFpsText = findViewById(R.id.image_classification_fps_text);
    mMsText = findViewById(R.id.image_classification_ms_text);
    mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);
  }

  @Override
  protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
    mMovingAvgSum += result.moduleForwardDuration;
    mMovingAvgQueue.add(result.moduleForwardDuration);
    if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
      mMovingAvgSum -= mMovingAvgQueue.remove();
    }

    for (int i = 0; i < TOP_K; i++) {
      final ResultRowView rowView = mResultRowViews[i];
      rowView.nameTextView.setText(result.topNClassNames[i]);
      rowView.scoreTextView.setText(String.format(Locale.US, SCORES_FORMAT,
          result.topNScores[i]));
      rowView.setProgressState(false);
    }

    mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
    if (mMsText.getVisibility() != View.VISIBLE) {
      mMsText.setVisibility(View.VISIBLE);
    }
    mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
    if (mFpsText.getVisibility() != View.VISIBLE) {
      mFpsText.setVisibility(View.VISIBLE);
    }

    if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
      float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
      mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
      if (mMsAvgText.getVisibility() != View.VISIBLE) {
        mMsAvgText.setVisibility(View.VISIBLE);
      }
    }
  }

  protected String getModuleAssetName() {
    if (!TextUtils.isEmpty(mModuleAssetName)) {
      return mModuleAssetName;
    }
    final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
    Log.d("INFO", moduleAssetNameFromIntent);
    mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
        ? moduleAssetNameFromIntent
        : "plateai_mobile.pt";

    return mModuleAssetName;
  }

  @Override
  protected String getInfoViewAdditionalText() {
    return getModuleAssetName();
  }

  @Override
  @WorkerThread
  @Nullable
  protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
    if (mAnalyzeImageErrorState) {
      return null;
    }

    try {
      if (mModule == null) {
        final String moduleFileAbsoluteFilePath = new File(
            Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
        mModule = Module.load(moduleFileAbsoluteFilePath);

        mInputTensorBuffer = Tensor.allocateFloatBuffer(1 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
        mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 1, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
      }

      final long startTime = SystemClock.elapsedRealtime();
      //TensorImageUtils.imageYUV420CenterCropToFloatBuffer(image.getImage(), rotationDegrees, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, mInputTensorBuffer, 0);

      // Without cropping, probably more flexible
      //TensorImageUtils.bitmapToFloatBuffer(toBitmap(image.getImage()), 0, 0, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, mInputTensorBuffer, 0);
      ImageProcessing.imageYUV420CenterCropToGrayscaleFloatBuffer(image.getImage(), rotationDegrees, INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, mInputTensorBuffer, 0);

      final long moduleForwardStartTime = SystemClock.elapsedRealtime();
      final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
      final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

      final float[] scores = outputTensor.getDataAsFloatArray();
      final List<List<Float>> charScores = new ArrayList<>();
      for(int i = 0; i < PRED_LENGTH; i++) {
        charScores.add(new ArrayList<>());
        for (int j = 0; j < ALPHABET.length(); j++) {
          charScores.get(i).add(scores[(i * ALPHABET.length()) + j]);
        }
      }

      List<PredConf> topK = getTopK(charScores, 0.01f);

      final String[] topKClassNames = new String[TOP_K];
      final float[] topKScores = new float[TOP_K];
      for (int i = 0; i < TOP_K; i++) {
        topKClassNames[i] = topK.get(i).pred; //decode(rawPred, rawConf);
        topKScores[i] = topK.get(i).conf; //100;
      }
      final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
      return new AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration);
    } catch (Exception e) {
      Log.e(Constants.TAG, "Error during image analysis", e);
      mAnalyzeImageErrorState = true;
      runOnUiThread(() -> {
        if (!isFinishing()) {
          showErrorDialog(v -> ImageClassificationActivity.this.finish());
        }
      });
      return null;
    }
  }

  @Override
  protected int getInfoViewCode() {
    return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (mModule != null) {
      mModule.destroy();
    }
  }

  protected String decode(List<String> rawPred, List<Float> rawConf) {
    String pred = "";
    for (int i = 0; i < rawPred.size(); i++) {
      String p = rawPred.get(i);

      if (rawConf != null) {
        Float c = rawConf.get(i);
        if (c < MIN_CONF) {
          p = BLANK;  // Replace low confidence characters with blank character
        }
      }

      if (pred.length() > 0 && p.equals(pred.substring(pred.length() - 1))) {
        // Do not add duplicates
      } else {
        pred += p;
      }
    }

    return pred.replace(BLANK,"").toUpperCase();
  }

  //@RequiresApi(api = Build.VERSION_CODES.N)
  private List<PredConf> getTopK(List<List<Float>> pred, Float minConf) {
    Map<String, Float> decPreds = new HashMap<>();
    List<Stuff> predStack = new ArrayList<>();

    List<List<Integer>> sortedIndexes = pred.stream().map(charProbs -> {
      List<ConfIdx> sortedConf = new ArrayList<>();
      for (int i=0; i<charProbs.size(); i++) {
        sortedConf.add(new ConfIdx(charProbs.get(i), i));
      }
      sort(sortedConf, new SortByConf());
      return sortedConf.stream().map(confIdx -> confIdx.idx).collect(toList());
    }).collect(toList());

    List<Integer> rawPred = sortedIndexes.stream().map(indexes -> indexes.remove(indexes.size() -1)).collect(toList());
    List<Float> rawConf = new ArrayList<>(); //zip(rawPred.stream(), IntStream.range(0, rawPred.size()).boxed(), (a,b) -> pred.get(b).get(a)).collect(toList());
    for (int i=0; i<rawPred.size(); i++){
      rawConf.add(pred.get(i).get(rawPred.get(i)));
    }

    PredConf maxPred = decodeAndStore(rawPred, rawConf, decPreds);

    float conf = maxPred.conf;
    predStack.add(new Stuff(rawPred, null, sortedIndexes, maxPred.conf, null, rawConf, null));

    while (decPreds.size() < TOP_K || conf >= minConf) {
      conf = topKStep(pred, predStack, decPreds);
    }

    List<PredConf> allPreds = decPreds.entrySet().stream()
            .map(e -> new PredConf(e.getKey(),e.getValue()))
            .sorted((a,b) -> Float.compare(b.conf, a.conf))
            .collect(toList());

    return allPreds.subList(0, TOP_K);
  }

  private float topKStep(List<List<Float>> pred, List<Stuff> predStack, Map<String, Float> decPreds) {
    Stuff cur = predStack.remove(predStack.size()-1);  // Sort reversed and remove(0)?
    topKSubStep(cur.rawPred, cur.rawConf, cur.sortedIdxs, pred, predStack);

    if (cur.newPred != null) {
      decodeAndStore(cur.newPred, cur.newRawConf, decPreds);
      topKSubStep(cur.newPred, cur.newRawConf, cur.sortedIdxs, pred, predStack);  // Use newConf instead
    }

    return cur.newConf != null ? cur.newConf : 0f;
  }

  private void topKSubStep(List<Integer> rawPred, List<Float> rawConf, List<List<Integer>> sortedIndexes, List<List<Float>> pred, List<Stuff> predStack) {
    Float minDiff = null;
    Integer minIndex = null;
    for (int i=0; i<pred.size(); i++) {
      int rawIndex = rawPred.get(i);
      List<Integer> indexes = sortedIndexes.get(i);
      List<Float> charPred = pred.get(i);
      Float confDiff = indexes.size() > 0 ? charPred.get(rawIndex)/charPred.get(indexes.get(indexes.size()-1)) : null;
      if (confDiff != null) {
        if (minDiff == null || confDiff < minDiff) {
          minDiff = confDiff;
          minIndex = i;
        }
      }
    }

    List<List<Integer>> sortedIdxs = new ArrayList<>(sortedIndexes);  // Sufficient copy?
    Integer newIndex = sortedIdxs.get(minIndex).remove(sortedIdxs.get(minIndex).size()-1);
    List<Integer> newPred = new ArrayList<>(rawPred);
    List<Float> newRawConf = new ArrayList<>(rawConf);
    newPred.set(minIndex, newIndex);
    newRawConf.set(minIndex, pred.get(minIndex).get(newIndex));
    float oldConf = rawConf.stream().reduce(1f, (a, b) -> a*b);
    float newConf = newRawConf.stream().reduce(1f, (a, b) -> a*b);

    boolean inserted = false;
    for (int i=0; i<predStack.size(); i++) {
      float otherConf = predStack.get(i).newConf;
      if (newConf < otherConf) {
        predStack.add(i, new Stuff(rawPred, newPred, sortedIdxs, oldConf, newConf, rawConf, newRawConf));
        inserted = true;
        break;
      }
    }

    if (!inserted) {
      predStack.add(new Stuff(rawPred, newPred, sortedIdxs, oldConf, newConf, rawConf, newRawConf));
    }
  }

  private PredConf decodeAndStore(List<Integer> rawPred, List<Float> rawConf, Map<String, Float> dict) {
    List<String> rawString = rawPred.stream().map(i -> "" + ALPHABET.charAt(i)).collect(toList());
    String decodedPred = decode(rawString, rawConf);
    float conf = rawConf.stream().reduce(1f, (a, b) -> a*b);
    float totalConf = conf + dict.getOrDefault(decodedPred, 0f);
    dict.put(decodedPred, totalConf);
    return new PredConf(decodedPred, conf);
  }

}
