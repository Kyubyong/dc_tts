package com.android.simsax;

import java.io.File;

public class ExampleRow {
    private int imageButton;
    private String audioName;
    private File file;

    public ExampleRow(int imageButton, String audioName, File file) {
        this.imageButton = imageButton;
        this.audioName = audioName;
        this.file = file;
    }

    public int getImageButton() {
        return imageButton;
    }

    public void setImageButton(int imageButton) {
        this.imageButton = imageButton;
    }

    public String getAudioName() {
        return audioName;
    }

    public void setAudioName(String name) {this.audioName = name;}

    public File getFile() {
        return file;
    }

}
