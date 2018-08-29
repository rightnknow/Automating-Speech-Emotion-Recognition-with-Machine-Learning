package project.ece496.speechsentiments.analysis;

import android.view.View;
import android.widget.TextView;

/**
 * An interface for tone analysis from text.
 */

public interface TextToneAnalyzer<T> {
    public T analyze(String text);
}
