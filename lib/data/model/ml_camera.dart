import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../entity/recognition.dart';
import 'classifier.dart';
import '../../util/image_utils.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:weed_app/data/entity/recognition.dart';

final recognitionsProvider = StateProvider<List<Recognition>>((ref) => []);
final mlCameraProvider =
    FutureProvider.autoDispose.family<MLCamera, Size>((ref, size) async {
  final cameras = await availableCameras();
  final cameraController = CameraController(
    cameras[0],
    ResolutionPreset.low,
    enableAudio: false,
  );
  await cameraController.initialize();
  final mlCamera = MLCamera(
    ref.read,
    cameraController,
    size,
  );
  return mlCamera;
});

class MLCamera {
  MLCamera(
    this._read,
    this.cameraController,
    this.cameraViewSize,
  ) {
    Future(() async {
      classifier = Classifier();
      ratio = Platform.isAndroid
          ? cameraViewSize.width / cameraController.value.previewSize!.height
          : cameraViewSize.width / cameraController.value.previewSize!.width;

      actualPreviewSize = Size(
        cameraViewSize.width,
        cameraViewSize.width * ratio,
      );
      await cameraController.startImageStream(onLatesImageAvailable);
    });
  }
  final Reader _read;
  final CameraController cameraController;

  Size cameraViewSize;
  double ratio;
  Classifier classifier;
  bool isPredicting = false;
  Size actualPreviewSize;

  Future<void> onLatesImageAvailable(CameraImage cameraImage) async {
    if (classifier.interpreter == null || classifier.labels == null) {
      return;
    }
    if (isPredicting) {
      return;
    }
    isPredicting = true;
    final isolateCamImageData = IsolateData(
      cameraImage: cameraImage,
      intrtpreterAddress: classifier.interpreter.address,
      labels: classifier.labels,
    );

    _read(recognitionsProvider).state =
        await compute(inference, isolateCamImageData);
    isPredicting = false;
  }

  static Future<List<Recognition>> inference(
      IsolateData isolateCamImageData) async {
    var image = ImageUtils.convertYUV420ToImage(
      isolateCamImageData.cameraImage,
    );
    if (Platform.isAndroid) {
      image = image_lib.copyRotate(image, 90);
    }

    final classifier = Classifier(
      interpreter: Interpreter.fromAddress(
        isolateCamImageData.intrtpreterAddress,
      ),
      labels: isolateCamImageData.labels,
    );
    return classifier.predict(image);
  }
}

class IsolateData {
  IsolateData({required this.cameraImage, required this.intrtpreterAddress, required this.labels});

  final CameraImage cameraImage;
  final int intrtpreterAddress;
  final List<String> labels;
}
