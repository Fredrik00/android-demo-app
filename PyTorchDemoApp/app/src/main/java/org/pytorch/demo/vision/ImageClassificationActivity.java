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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import static java.util.Collections.*;
import static java.util.stream.Collectors.toList;

public class ImageClassificationActivity extends AbstractCameraXActivity<ImageClassificationActivity.AnalysisResult> {

  public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
  public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

  private static final int INPUT_TENSOR_WIDTH = 100;
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
  private static final double MIN_CONF = 0.5;

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
        : "plateai.pt";

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

      /*
      final List<String> rawPred = new ArrayList<>();
      final List<Float> rawConf = new ArrayList<>();

      // Naive max at each character spot
      for (List<Float> charScore : charScores) {
        int maxIdx = 0;
        float maxVal = 0;
        int i = 0;
        for (Float score : charScore) {
          if (score > maxVal) {
            maxIdx = i;
            maxVal = score;
          }
          i += 1;
        }
        rawPred.add("" + ALPHABET.charAt(maxIdx));
        rawConf.add(maxVal);
      }
       */
      PredConf bestPred = getTopN(5, charScores, 0.01f).get(0);

      final String[] topKClassNames = new String[TOP_K];
      final float[] topKScores = new float[TOP_K];
      for (int i = 0; i < TOP_K; i++) {
        topKClassNames[i] = bestPred.pred; //decode(rawPred, rawConf);
        topKScores[i] = bestPred.conf; //100;
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
      Float c = rawConf.get(i);
      if (c < MIN_CONF) {
        p = BLANK;  // Replace low confidence characters with blank character
      }

      if (pred.length() > 0 && p.equals(pred.substring(pred.length() - 1))) {
        // Do not add duplicates
      } else {
        pred += p;
      }
    }

    return pred.replace(BLANK,"").toUpperCase();
  }

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

    public Stuff(List<Integer> rawPred, List<Integer> newPred, List<List<Integer>> sortedIdxs, Float conf, Float newConf) {
      this.rawPred = rawPred;
      this.newPred = newPred;
      this.sortedIdxs = sortedIdxs;
      this.conf = conf;
      this.newConf = newConf;
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

  //@RequiresApi(api = Build.VERSION_CODES.N)
  private List<PredConf> getTopN(int n, List<List<Float>> pred, Float minConf) {
    HashMap<String, Float> decPreds = new HashMap<>();
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

    // TODO: Replace when done
    PredConf rawMax = new PredConf(decode(rawPred.stream().map(i -> "" + ALPHABET.charAt(i)).collect(toList()), rawConf), rawConf.stream().reduce(1f, (a, b) -> a*b));
    List<PredConf> topN = Collections.singletonList(rawMax);
    return topN;
  }

  /*
    # Keep track of decoded prediction with confidence, and add starting point to stack to prepare the loop
    add_decoded_pred_to_dict(raw_pred, raw_conf, dec_preds, converter)
    pred_stack.append((raw_pred, None, sorted_indexes, raw_conf, None))

    # Iteratively choose the highest confidence raw prediction until we have n decoded predictions and only low confidence raw predictions remain
    while len(dec_preds) < n or raw_conf >= min_conf:
        raw_conf = top_n_step(pred, pred_stack, dec_preds, converter)

    # We sort all decoded strings by their confidence and select the n best ones
    dec_pred_list = [[string, prob] for string, prob in dec_preds.items()]
    dec_pred_list.sort(key=operator.itemgetter(1), reverse=True)
    return dec_pred_list[:n]


def top_n_step(pred, pred_stack, dec_preds, converter):
    # Select our next starting point from the top of the stack
    # best_index = np.argmax([raw_conf for _, _, raw_conf in pred_stack])
    raw_pred, new_pred, sorted_indexes, raw_conf, new_conf = pred_stack.pop()  # pop using best_index if not sorting
    top_n_substep(raw_pred, raw_conf, sorted_indexes, pred, pred_stack)

    if new_pred is not None:
        add_decoded_pred_to_dict(new_pred, new_conf, dec_preds, converter)
        top_n_substep(new_pred, new_conf, sorted_indexes, pred, pred_stack)

    return new_conf


def top_n_substep(raw_pred, raw_conf, sorted_indexes, pred, pred_stack):
    conf_diffs = [char_pred[raw_index] / char_pred[indexes[-1]] if len(indexes) > 0 else np.nan for
                  raw_index, indexes, char_pred in zip(raw_pred, sorted_indexes, pred)]

    # Replace the character producing the lowest decrease in confidence (least ratio between character confidences)
    min_index = np.nanargmin(conf_diffs)

    # Copy sorted indexes to not limit choices for the next iteration
    sorted_idxs = sorted_indexes[:]
    new_index = sorted_idxs[min_index].pop()
    # Alter a copy of the raw string
    new_pred = raw_pred[:]
    new_pred[min_index] = new_index
    new_conf = np.prod([pred[i][new_index] for new_index, i in zip(new_pred, range(len(new_pred)))])

    # Insertion sort, should replace with binary insertion sort and maybe use blist
    inserted = False
    for i, (_, _, _, _, other_conf) in enumerate(pred_stack):
        if new_conf < other_conf:
            pred_stack.insert(i, (raw_pred, new_pred, sorted_idxs, raw_conf, new_conf))
            inserted = True
            break

    if not inserted:
        pred_stack.append((raw_pred, new_pred, sorted_idxs, raw_conf, new_conf))


def add_decoded_pred_to_dict(pred, confidence, sim_dict, converter):
    sim_string = converter.decode_single(pred, len(pred))
    if sim_dict.get(sim_string) is None:
        sim_dict[sim_string] = confidence
    else:
        sim_dict[sim_string] += confidence
   */
}
