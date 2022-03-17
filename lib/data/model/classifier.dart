import 'dart:math';
import '../entity/recognition.dart';
import '../../util/logger.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class Classifier {
  Classifier({
     Interpreter? interpreter,
     List<String>? labels,
  }) {
    loadModel(interpreter!);
    loadLabels(labels!);
  }

  Interpreter _interpreter;
  Interpreter get interpreter => _interpreter;
  List<String> _labels;
  List<String> get labels => _labels;
  static const String modelFileName = 'yolov5m-fp16.tflite';
  static const String labelFileName = 'yolov5m-fp16.txt';

  static const int inputSize = 300;
  static const double threshold = 0.6;
  ImageProcessor imageProcessor;
  List<List<int>> _outputShapes;
  List<TfLiteType> _outputTypes;
  static const int numResult = 10;

  Future<void> loadModel(Interpreter interpreter) async {
    try {
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            '$modelFileName',
            options: InterpreterOptions()
              ..threads = 4,
          );
      final outputTensors = _interpreter.getInputTensors();
      _outputShapes = [];
      _outputTypes = [];
      for (final tensor in outputTensors) {
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
      }
    } on Exception catch (e) {
      logger.warning(e.toString());
    }
  }

  Future<void> loadLabels(List<String> labels) async {
    try {
      _labels = labels ?? await FileUtil.loadLabels('assets/$labelFileName');
    } on Exception catch (e) {
      logger.warning(e);
    }
  }

  TensorImage getProcessedImage(TensorImage inputImage) {
    final padSize = max(
      inputImage.height,
      inputImage.width,
    );
    imageProcessor ??= ImageProcessorBuilder()
        .add(
          ResizeWithCropOrPadOp(
            padSize,
            padSize,
          ),
        )
        .add(
          ResizeOp(
            inputSize,
            inputSize,
            ResizeMethod.BILINEAR,
          ),
        )
        .build();
    return imageProcessor.process(inputImage);
  }

  List<Recognition>? predict(image_lib.Image image) {
    if (_interpreter == null) {
      return null;
    }

    var inputImage = TensorImage.fromImage(image);
    inputImage = getProcessedImage(inputImage);

    final outputLocations = TensorBufferFloat(_outputShapes[0]);
    final outputClasses = TensorBufferFloat(_outputShapes[1]);
    final outputScores = TensorBufferFloat(_outputShapes[2]);
    final numLocations = TensorBufferFloat(_outputShapes[3]);

    final inputs = [inputImage.buffer];
    final outputs = {
      0: outputLocations.buffer,
      1: outputClasses.buffer,
      2: outputScores.buffer,
      3: numLocations.buffer,
    };

    _interpreter.runForMultipleInputs(inputs, outputs);
    final resultCount = min(numResult, numLocations.getIntValue(0));
    const labelOffset = 1;
    final locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      valueIndex: [0, 1, 2, 3],
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.BOUNDARIES,
      coordinateType: CoordinateType.RATIO,
      height: inputSize,
      width: inputSize,
    );

    final recognition = <Recognition>[];
    for (var i = 0; i < resultCount; i++) {
      final score = outputScores.getDoubleValue(i);
      final labelIndex = outputClasses.getIntValue(i) + labelOffset;
      final label = _labels.elementAt(labelIndex);
      if (score > threshold) {
        final transFormRect = imageProcessor.inverseTransformRect(
          locations[i],
          image.height,
          image.width,
        );
        recognition.add(
          Recognition(i, label, score, transFormRect),
        );
      }
    }
    return recognition;
  }
}
