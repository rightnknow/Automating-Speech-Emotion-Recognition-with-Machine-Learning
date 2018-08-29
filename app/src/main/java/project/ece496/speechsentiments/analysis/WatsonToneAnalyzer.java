package project.ece496.speechsentiments.analysis;

import com.ibm.watson.developer_cloud.tone_analyzer.v3.ToneAnalyzer;
import com.ibm.watson.developer_cloud.tone_analyzer.v3.model.ToneAnalysis;
import com.ibm.watson.developer_cloud.tone_analyzer.v3.model.ToneOptions;

import project.ece496.speechsentiments.BuildConfig;


/**
 * A class that wraps IBM Watson's Tone Analyzer
 */

public class WatsonToneAnalyzer implements TextToneAnalyzer<ToneAnalysis> {

    private ToneAnalyzer toneAnalyzer;

    public WatsonToneAnalyzer() {
        toneAnalyzer = new ToneAnalyzer(
                BuildConfig.ToneAnalyzerVersion,
                BuildConfig.ToneAnalyzerUsername,
                BuildConfig.ToneAnalyzerPassword);
    }

    public ToneAnalysis analyze(String text){
        ToneOptions toneOptions = new ToneOptions.Builder()
                .text(text)
                .build();

        ToneAnalysis toneAnalysis = toneAnalyzer.tone(toneOptions).execute();

        return toneAnalysis;
    }
}
