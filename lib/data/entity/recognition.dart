// @dart=2.9
import 'package:flutter/material.dart';
import 'dart:math';

class Recognition {
  Recognition(this._id, this._label, this._score, [this._location]);
  final int _id;
  int get id => _id;
  final String _label;
  String get label => _label;
  final double _score;
  double get score => _score;
  final Rect _location;
  Rect get location => _location;

  Rect getRenderLocation(Size actualPreviewSize, double pixelRatio) {
    final ratioX = pixelRatio;
    final ratioY = ratioX;

    final transLeft = max(0.1, location.left * ratioX);
    final transTop = max(0.1, location.top * ratioY);
    final transWidth = min(
      location.width * ratioX,
      actualPreviewSize.width,
    );
    final transHeight = min(
      location.height * ratioY,
      actualPreviewSize.height,
    );
    final transFormedRect =
        Rect.fromLTWH(transLeft, transTop, transWidth, transHeight);
    return transFormedRect;
  }
}
