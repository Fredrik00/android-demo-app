package org.pytorch.demo.vision;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Locale;

public final class ImageProcessing {

    public static void imageYUV420CenterCropToFloatBuffer(
            final Image image,
            int rotateCWDegrees,
            final int tensorWidth,
            final int tensorHeight,
            float[] normMeanRGB,
            float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {
        checkOutBufferCapacity(outBuffer, outBufferOffset, tensorWidth, tensorHeight);

        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException(
                    String.format(Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
        }

        checkNormMeanArg(normMeanRGB);
        checkNormStdArg(normStdRGB);
        checkRotateCWDegrees(rotateCWDegrees);
        checkTensorSize(tensorWidth, tensorHeight);

        final int widthBeforeRotation = image.getWidth();
        final int heightBeforeRotation = image.getHeight();

        int widthAfterRotation = widthBeforeRotation;
        int heightAfterRotation = heightBeforeRotation;
        if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
            widthAfterRotation = heightBeforeRotation;
            heightAfterRotation = widthBeforeRotation;
        }

        int centerCropWidthAfterRotation = widthAfterRotation;
        int centerCropHeightAfterRotation = heightAfterRotation;

        if (tensorWidth * heightAfterRotation <= tensorHeight * widthAfterRotation) {
            centerCropWidthAfterRotation =
                    (int) Math.floor((float) tensorWidth * heightAfterRotation / tensorHeight);
        } else {
            centerCropHeightAfterRotation =
                    (int) Math.floor((float) tensorHeight * widthAfterRotation / tensorWidth);
        }

        int centerCropWidthBeforeRotation = centerCropWidthAfterRotation;
        int centerCropHeightBeforeRotation = centerCropHeightAfterRotation;
        if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
            centerCropHeightBeforeRotation = centerCropWidthAfterRotation;
            centerCropWidthBeforeRotation = centerCropHeightAfterRotation;
        }

        final int offsetX =
                (int) Math.floor((widthBeforeRotation - centerCropWidthBeforeRotation) / 2.f);
        final int offsetY =
                (int) Math.floor((heightBeforeRotation - centerCropHeightBeforeRotation) / 2.f);

        final Image.Plane yPlane = image.getPlanes()[0];  // Should be the grayscale
        final Image.Plane uPlane = image.getPlanes()[1];
        final Image.Plane vPlane = image.getPlanes()[2];

        final ByteBuffer yBuffer = yPlane.getBuffer();
        final ByteBuffer uBuffer = uPlane.getBuffer();
        final ByteBuffer vBuffer = vPlane.getBuffer();

        final int yRowStride = yPlane.getRowStride();
        final int uRowStride = uPlane.getRowStride();

        final int yPixelStride = yPlane.getPixelStride();
        final int uPixelStride = uPlane.getPixelStride();

        final float scale = (float) centerCropWidthAfterRotation / tensorWidth;
        final int uvRowStride = uRowStride >> 1;

        final int channelSize = tensorHeight * tensorWidth;
        final int tensorInputOffsetG = channelSize;
        final int tensorInputOffsetB = 2 * channelSize;
        for (int x = 0; x < tensorWidth; x++) {
            for (int y = 0; y < tensorHeight; y++) {

                final int centerCropXAfterRotation = (int) Math.floor(x * scale);
                final int centerCropYAfterRotation = (int) Math.floor(y * scale);

                int xBeforeRotation = offsetX + centerCropXAfterRotation;
                int yBeforeRotation = offsetY + centerCropYAfterRotation;
                if (rotateCWDegrees == 90) {
                    xBeforeRotation = offsetX + centerCropYAfterRotation;
                    yBeforeRotation =
                            offsetY + (centerCropHeightBeforeRotation - 1) - centerCropXAfterRotation;
                } else if (rotateCWDegrees == 180) {
                    xBeforeRotation =
                            offsetX + (centerCropWidthBeforeRotation - 1) - centerCropXAfterRotation;
                    yBeforeRotation =
                            offsetY + (centerCropHeightBeforeRotation - 1) - centerCropYAfterRotation;
                } else if (rotateCWDegrees == 270) {
                    xBeforeRotation =
                            offsetX + (centerCropWidthBeforeRotation - 1) - centerCropYAfterRotation;
                    yBeforeRotation = offsetY + centerCropXAfterRotation;
                }

                final int yIdx = yBeforeRotation * yRowStride + xBeforeRotation * yPixelStride;
                final int uvIdx = (yBeforeRotation >> 1) * uvRowStride + xBeforeRotation * uPixelStride;

                int Yi = yBuffer.get(yIdx) & 0xff;
                int Ui = uBuffer.get(uvIdx) & 0xff;
                int Vi = vBuffer.get(uvIdx) & 0xff;

                int a0 = 1192 * (Yi - 16);
                int a1 = 1634 * (Vi - 128);
                int a2 = 832 * (Vi - 128);
                int a3 = 400 * (Ui - 128);
                int a4 = 2066 * (Ui - 128);

                int r = clamp((a0 + a1) >> 10, 0, 255);
                int g = clamp((a0 - a2 - a3) >> 10, 0, 255);
                int b = clamp((a0 + a4) >> 10, 0, 255);
                final int offset = outBufferOffset + y * tensorWidth + x;
                float rF = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
                float gF = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
                float bF = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

                outBuffer.put(offset, rF);
                outBuffer.put(offset + tensorInputOffsetG, gF);
                outBuffer.put(offset + tensorInputOffsetB, bF);
            }
        }
    }

    public static void imageYUV420CenterCropToGrayscaleFloatBuffer(
            final Image image,
            int rotateCWDegrees,
            final int tensorWidth,
            final int tensorHeight,
            float[] normMeanRGB,
            float[] normStdRGB,
            final FloatBuffer outBuffer,
            final int outBufferOffset) {
        //checkOutBufferCapacity(outBuffer, outBufferOffset, tensorWidth, tensorHeight);

        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException(String.format(Locale.US, "Image format %d != ImageFormat.YUV_420_888", image.getFormat()));
        }

        checkNormMeanArg(normMeanRGB);
        checkNormStdArg(normStdRGB);
        checkRotateCWDegrees(rotateCWDegrees);
        checkTensorSize(tensorWidth, tensorHeight);

        final int widthBeforeRotation = image.getWidth();
        final int heightBeforeRotation = image.getHeight();
        Log.d("width", "" + widthBeforeRotation);
        Log.d("height", "" + heightBeforeRotation);

        int widthAfterRotation = widthBeforeRotation;
        int heightAfterRotation = heightBeforeRotation;
        if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
            widthAfterRotation = heightBeforeRotation;
            heightAfterRotation = widthBeforeRotation;
        }

        int centerCropWidthAfterRotation = widthAfterRotation;
        int centerCropHeightAfterRotation = heightAfterRotation;

        if (tensorWidth * heightAfterRotation <= tensorHeight * widthAfterRotation) {
            centerCropWidthAfterRotation = (int) Math.floor((float) tensorWidth * heightAfterRotation / tensorHeight);
        } else {
            centerCropHeightAfterRotation = (int) Math.floor((float) tensorHeight * widthAfterRotation / tensorWidth);
        }

        int centerCropWidthBeforeRotation = centerCropWidthAfterRotation;
        int centerCropHeightBeforeRotation = centerCropHeightAfterRotation;
        if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
            centerCropHeightBeforeRotation = centerCropWidthAfterRotation;
            centerCropWidthBeforeRotation = centerCropHeightAfterRotation;
        }

        final int offsetX = (int) Math.floor((widthBeforeRotation - centerCropWidthBeforeRotation) / 2.f);
        final int offsetY = (int) Math.floor((heightBeforeRotation - centerCropHeightBeforeRotation) / 2.f);

        final Image.Plane yPlane = image.getPlanes()[0];  // Should be the grayscale

        final ByteBuffer yBuffer = yPlane.getBuffer();

        final int yRowStride = yPlane.getRowStride();

        final int yPixelStride = yPlane.getPixelStride();

        final float scale = (float) centerCropWidthAfterRotation / tensorWidth;

        final int channelSize = tensorHeight * tensorWidth;

        for (int x = 0; x < tensorWidth; x++) {
            for (int y = 0; y < tensorHeight; y++) {

                final int centerCropXAfterRotation = (int) Math.floor(x * scale);
                final int centerCropYAfterRotation = (int) Math.floor(y * scale);

                int xBeforeRotation = offsetX + centerCropXAfterRotation;
                int yBeforeRotation = offsetY + centerCropYAfterRotation;
                if (rotateCWDegrees == 90) {
                    xBeforeRotation = offsetX + centerCropYAfterRotation;
                    yBeforeRotation = offsetY + (centerCropHeightBeforeRotation - 1) - centerCropXAfterRotation;
                } else if (rotateCWDegrees == 180) {
                    xBeforeRotation = offsetX + (centerCropWidthBeforeRotation - 1) - centerCropXAfterRotation;
                    yBeforeRotation = offsetY + (centerCropHeightBeforeRotation - 1) - centerCropYAfterRotation;
                } else if (rotateCWDegrees == 270) {
                    xBeforeRotation = offsetX + (centerCropWidthBeforeRotation - 1) - centerCropYAfterRotation;
                    yBeforeRotation = offsetY + centerCropXAfterRotation;
                }

                final int yIdx = yBeforeRotation * yRowStride + xBeforeRotation * yPixelStride;

                int Yi = yBuffer.get(yIdx) & 0xff;

                //int a0 = 1192 * (Yi - 16);
                // assuming Yi between 0 and 255?

                // final int offset = outBufferOffset + y * tensorWidth + x;
                // float rF = ((Yi / 255.f) - normMeanRGB[0]) / normStdRGB[0];
                //float gF = ((Yi / 255.f) - normMeanRGB[1]) / normStdRGB[1];
                //float bF = ((Yi / 255.f) - normMeanRGB[2]) / normStdRGB[2];

                //outBuffer.put(offset, rF);
                //outBuffer.put(offset + channelSize, gF);
                //outBuffer.put(offset + 2*channelSize, bF);

                final int offset = outBufferOffset + y*tensorWidth+x;
                outBuffer.put(offset, Yi / 255.f);
            }
        }
    }

    private static void checkOutBufferCapacity(FloatBuffer outBuffer, int outBufferOffset, int tensorWidth, int tensorHeight) {
        if (outBufferOffset + 3 * tensorWidth * tensorHeight > outBuffer.capacity()) {
            throw new IllegalStateException("Buffer underflow");
        }
    }

    private static void checkTensorSize(int tensorWidth, int tensorHeight) {
        if (tensorHeight <= 0 || tensorWidth <= 0) {
            throw new IllegalArgumentException("tensorHeight and tensorWidth must be positive");
        }
    }

    private static void checkRotateCWDegrees(int rotateCWDegrees) {
        if (rotateCWDegrees != 0
                && rotateCWDegrees != 90
                && rotateCWDegrees != 180
                && rotateCWDegrees != 270) {
            throw new IllegalArgumentException("rotateCWDegrees must be one of 0, 90, 180, 270");
        }
    }

    private static final int clamp(int c, int min, int max) {
        return c < min ? min : c > max ? max : c;
    }

    private static void checkNormStdArg(float[] normStdRGB) {
        if (normStdRGB.length != 3) {
            throw new IllegalArgumentException("normStdRGB length must be 3");
        }
    }

    private static void checkNormMeanArg(float[] normMeanRGB) {
        if (normMeanRGB.length != 3) {
            throw new IllegalArgumentException("normMeanRGB length must be 3");
        }
    }

    private Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();  // Should be the grayscale
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(49, 7, 273, 231), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
}
