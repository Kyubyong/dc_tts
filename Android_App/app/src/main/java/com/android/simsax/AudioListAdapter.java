package com.android.simsax;

import android.app.Activity;
import android.graphics.Color;
import android.view.ActionMode;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.FragmentActivity;
import androidx.lifecycle.LifecycleOwner;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;
import androidx.recyclerview.widget.RecyclerView;
import java.util.ArrayList;

public class AudioListAdapter extends RecyclerView.Adapter<AudioListAdapter.AudioViewHolder> {

    private ArrayList<ExampleRow> exampleList;
    private OnItemClickListener mListener;
    private boolean isEnable = false;
    private boolean allSelected = false;
    private ArrayList<Boolean> selected = new ArrayList<>();
    private ArrayList<ExampleRow> selectedItems = new ArrayList<>();
    private MainViewModel mainViewModel;
    private Activity activity;


    public class AudioViewHolder extends RecyclerView.ViewHolder {

        public ImageView buttonPlay;
        public TextView fileName;
        public ImageView shareButton;

        public AudioViewHolder(@NonNull View itemView, OnItemClickListener listener) {
            super(itemView);

            buttonPlay = itemView.findViewById(R.id.button1);
            fileName = itemView.findViewById(R.id.fileName1);
            shareButton = itemView.findViewById(R.id.share_button);

            itemView.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if (listener != null) {
                        if (!isEnable) {
                            int position = getAdapterPosition();
                            if (position != RecyclerView.NO_POSITION) {
                                listener.onItemClick(position);
                            }
                        } else { // Action mode
                            ClickItem(itemView, getAdapterPosition());
                        }
                    }
                }
            });

            shareButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    if (listener != null) {
                        if (!isEnable) {
                            int position = getAdapterPosition();
                            if (position != RecyclerView.NO_POSITION) {
                                listener.onShareClick(position);
                            }
                        } else { // Action mode
                            ClickItem(itemView, getAdapterPosition());
                        }
                    }
                }
            });
        }
    }

    public AudioListAdapter(Activity activity, ArrayList<ExampleRow> exampleList) {
        this.exampleList = exampleList;
        this.activity = activity;
        for (int i=0; i<exampleList.size(); i++)
            selected.add(false);
    }

    @Override
    public AudioViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.example_row, parent, false);
        mainViewModel = ViewModelProviders.of((FragmentActivity) activity).get(MainViewModel.class);
        return new AudioViewHolder(view, mListener);
    }

    @Override
    public void onBindViewHolder(@NonNull AudioViewHolder holder, int position) {
        ExampleRow currentItem = exampleList.get(position);

        holder.fileName.setText(currentItem.getAudioName());
        holder.buttonPlay.setImageResource(currentItem.getImageButton());

        holder.itemView.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {

                if (!isEnable) {
                    ActionMode.Callback callback = new ActionMode.Callback() {
                        @Override
                        public boolean onCreateActionMode(ActionMode mode, Menu menu) {
                            MenuInflater menuInflater = mode.getMenuInflater();
                            menuInflater.inflate(R.menu.menu_select, menu);
                            return true;
                        }

                        @Override
                        public boolean onPrepareActionMode(ActionMode mode, Menu menu) {
                            isEnable = true;
                            ClickItem(holder.itemView, holder.getAdapterPosition());
                            mainViewModel.getText().observe((LifecycleOwner) activity,
                                    new Observer<String>() {
                                        @Override
                                        public void onChanged(String s) {
                                            if (s.equals("0"))
                                                mode.finish();
                                            else
                                                mode.setTitle(String.format("%s", s));
                                        }
                                    });
                            return true;
                        }

                        @Override
                        public boolean onActionItemClicked(ActionMode mode, MenuItem item) {
                            int id = item.getItemId();

                            if (id == R.id.menu_delete) {
                                for (ExampleRow f : selectedItems) {
                                    exampleList.remove(f);
                                    f.getFile().delete();
                                }
                                mode.finish();
                            } else {
                                if (selectedItems.size() == exampleList.size())  {
                                    selectedItems.clear();
                                    for (int i=0; i<selected.size(); i++)
                                        selected.set(i, false);
                                    allSelected = false;
                                    mode.finish();
                                } else {
                                    selectedItems.clear();
                                    selectedItems.addAll(exampleList);
                                    for (int i=0; i<selected.size(); i++)
                                        selected.set(i, true);
                                    allSelected = true;
                                    mode.setTitle(String.format("%d", selectedItems.size()));
                                }
                                notifyDataSetChanged();
                            }
                            return true;
                        }

                        @Override
                        public void onDestroyActionMode(ActionMode mode) {
                            isEnable = false;
                            allSelected = false;
                            selectedItems.clear();
                            for (int i=0; i < selected.size(); i++)
                                selected.set(i, false);
                            notifyDataSetChanged();
                        }
                    };

                    ((AppCompatActivity) v.getContext()).startActionMode(callback);
                } else {
                    ClickItem(holder.itemView, holder.getAdapterPosition());
                }
                return true;
            }
        });
        if (allSelected) {
            holder.itemView.setBackgroundColor(0xffb7f7f5);
        } else {
            holder.itemView.setBackgroundColor(Color.WHITE);
        }
    }

    private void ClickItem(View itemView, int position) {
        // get selected item
        ExampleRow selectedItem = exampleList.get(position);
        if (!selected.get(position)) {
            selected.set(position, true);
            itemView.setBackgroundColor(0xffb7f7f5);
            selectedItems.add(selectedItem);
        } else {
            selected.set(position, false);
            itemView.setBackgroundColor(Color.WHITE);
            selectedItems.remove(selectedItem);
        }
        mainViewModel.setText(String.valueOf(selectedItems.size()));
    }

    @Override
    public int getItemCount() {
        return exampleList.size();
    }

    public interface OnItemClickListener {
        void onItemClick(int position);
        void onShareClick(int position);
    }

    public void setOnItemClickListener(OnItemClickListener listener) {
        mListener = listener;
    }

}
