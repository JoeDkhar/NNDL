"""
Lab Assignment 04: Implementing CNN on the Intel Image Classification Dataset
Author: Josaiah
Roll No: 125
Dataset: Intel Image Classification (Natural Scenes)
Categories: buildings, forest, glacier, mountain, sea, street
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class IntelImageClassifier:
    """
    CNN Model for Intel Image Classification
    """
    
    def __init__(self, img_height=150, img_width=150, batch_size=32):
        """
        Initialize the classifier with image dimensions and batch size
        """
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        self.num_classes = len(self.class_names)
        self.model = None
        self.history = None
        
        # Define paths
        self.train_dir = 'seg_train/seg_train'
        self.test_dir = 'seg_test/seg_test'
        self.pred_dir = 'seg_pred/seg_pred'
        
    def load_and_explore_data(self):
        """
        Task 1: Dataset Overview - Visualize samples from the dataset
        """
        print("="*60)
        print("TASK 1: DATASET OVERVIEW")
        print("="*60)
        
        # Count images in each category
        print("\nDataset Statistics:")
        print("-" * 60)
        
        for dataset_name, dataset_path in [('Training', self.train_dir), 
                                           ('Testing', self.test_dir)]:
            print(f"\n{dataset_name} Set:")
            total_images = 0
            for class_name in self.class_names:
                class_path = os.path.join(dataset_path, class_name)
                if os.path.exists(class_path):
                    num_images = len([f for f in os.listdir(class_path) 
                                     if f.endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  {class_name}: {num_images} images")
                    total_images += num_images
            print(f"  Total: {total_images} images")
        
        # Visualize sample images from each category
        self._visualize_samples()
        
    def _visualize_samples(self):
        """
        Visualize sample images from each category
        """
        print("\nVisualizing sample images from each category...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Sample Images from Intel Image Classification Dataset', 
                     fontsize=16, fontweight='bold')
        
        for idx, class_name in enumerate(self.class_names):
            row = idx // 3
            col = idx % 3
            
            class_path = os.path.join(self.train_dir, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    # Load and display a random image
                    img_path = os.path.join(class_path, np.random.choice(images))
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    axes[row, col].imshow(img)
                    axes[row, col].set_title(f'{class_name.capitalize()} (Label: {idx})', 
                                            fontsize=12, fontweight='bold')
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
        print("Sample images saved as 'dataset_samples.png'")
        plt.show()
        
    def create_data_generators(self, use_augmentation=True):
        """
        Task 3: Create data generators with optional augmentation
        Task 5: Implement data augmentation
        """
        print("\n" + "="*60)
        print("TASK 3: DATA PREPARATION")
        print("="*60)
        
        # Training data generator with augmentation
        if use_augmentation:
            print("\nApplying data augmentation to training set...")
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,          # Random rotation
                width_shift_range=0.2,      # Random horizontal shift
                height_shift_range=0.2,     # Random vertical shift
                shear_range=0.2,            # Shear transformation
                zoom_range=0.2,             # Random zoom
                horizontal_flip=True,        # Random horizontal flip
                fill_mode='nearest',
                validation_split=0.2         # 20% for validation
            )
        else:
            print("\nNo data augmentation applied...")
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2
            )
        
        # Test data generator (only rescaling, no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Create validation generator
        self.validation_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Create test generator
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"\nTraining samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Batch size: {self.batch_size}")
        
    def build_cnn_model(self):
        """
        Task 2: Design CNN model with multiple convolutional layers
        Implements batch normalization and dropout
        """
        print("\n" + "="*60)
        print("TASK 2: CNN MODEL ARCHITECTURE")
        print("="*60)
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(256, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        
        print("\nModel Architecture Summary:")
        print("-" * 60)
        self.model.summary()
        
        # Save model architecture visualization
        print("\nModel architecture details printed above.")
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Task 3: Compile the model with appropriate loss function and optimizer
        """
        print("\n" + "="*60)
        print("MODEL COMPILATION")
        print("="*60)
        
        # Use Adam optimizer with custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nModel compiled successfully!")
        print(f"Optimizer: Adam (learning_rate={learning_rate})")
        print(f"Loss function: categorical_crossentropy")
        print(f"Metrics: accuracy")
        
    def train_model(self, epochs=50):
        """
        Task 3: Train the model with callbacks
        """
        print("\n" + "="*60)
        print("TASK 3: MODEL TRAINING")
        print("="*60)
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        callbacks = [early_stopping, reduce_lr, model_checkpoint]
        
        print(f"\nTraining for {epochs} epochs with callbacks:")
        print("  - Early Stopping (patience=10)")
        print("  - Learning Rate Reduction (patience=5)")
        print("  - Model Checkpoint (save best model)")
        print("\nTraining started...")
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        return self.history
    
    def plot_training_history(self):
        """
        Task 4: Plot training and validation accuracy/loss curves
        """
        print("\n" + "="*60)
        print("TASK 4: TRAINING VISUALIZATION")
        print("="*60)
        
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot training & validation loss
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\nTraining history plots saved as 'training_history.png'")
        plt.show()
        
        # Print final metrics
        print("\nFinal Training Metrics:")
        print("-" * 60)
        print(f"Training Accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"Validation Accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        print(f"Training Loss: {self.history.history['loss'][-1]:.4f}")
        print(f"Validation Loss: {self.history.history['val_loss'][-1]:.4f}")
        
    def evaluate_model(self):
        """
        Task 4: Evaluate the model on test set and display confusion matrix
        """
        print("\n" + "="*60)
        print("TASK 4: MODEL EVALUATION")
        print("="*60)
        
        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_loss, test_accuracy = self.model.evaluate(self.test_generator, verbose=1)
        
        print(f"\nTest Set Results:")
        print("-" * 60)
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Get predictions
        print("\nGenerating predictions for confusion matrix...")
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        # Calculate and display confusion matrix
        self._plot_confusion_matrix(true_classes, predicted_classes)
        
        # Display classification report
        print("\nClassification Report:")
        print("-" * 60)
        print(classification_report(true_classes, predicted_classes, 
                                   target_names=self.class_names))
        
        return test_accuracy, test_loss
    
    def _plot_confusion_matrix(self, true_labels, predicted_labels):
        """
        Task 4: Display confusion matrix
        """
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Intel Image Classification', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
        # Calculate per-class accuracy
        print("\nPer-Class Accuracy:")
        print("-" * 60)
        for i, class_name in enumerate(self.class_names):
            class_accuracy = cm[i, i] / cm[i].sum()
            print(f"{class_name.capitalize()}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    def predict_samples(self, num_samples=10):
        """
        Predict and visualize random samples from test set
        """
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)
        
        # Get random samples
        self.test_generator.reset()
        sample_images, sample_labels = next(self.test_generator)
        
        # Select random indices
        indices = np.random.choice(len(sample_images), 
                                  min(num_samples, len(sample_images)), 
                                  replace=False)
        
        # Make predictions
        predictions = self.model.predict(sample_images[indices])
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(sample_labels[indices], axis=1)
        
        # Visualize predictions
        num_cols = 5
        num_rows = (len(indices) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (img_idx, ax) in enumerate(zip(indices, axes.flatten())):
            if idx < len(indices):
                img = sample_images[img_idx]
                true_label = self.class_names[true_classes[idx]]
                pred_label = self.class_names[predicted_classes[idx]]
                confidence = predictions[idx][predicted_classes[idx]] * 100
                
                ax.imshow(img)
                color = 'green' if true_classes[idx] == predicted_classes[idx] else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                           fontsize=10, color=color, fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        print(f"\nSample predictions saved as 'sample_predictions.png'")
        plt.show()
    
    def save_model(self, filepath='intel_image_classifier.h5'):
        """
        Save the trained model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"\nModel saved as '{filepath}'")
        else:
            print("No model to save. Train the model first.")
    
    def load_model(self, filepath='intel_image_classifier.h5'):
        """
        Load a trained model
        """
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"\nModel loaded from '{filepath}'")
        else:
            print(f"Model file '{filepath}' not found.")


def main():
    """
    Main function to execute the complete pipeline
    """
    print("\n" + "="*60)
    print("INTEL IMAGE CLASSIFICATION WITH CNN")
    print("Lab Assignment 04 - Josaiah (125)")
    print("="*60)
    
    # Initialize classifier
    classifier = IntelImageClassifier(img_height=150, img_width=150, batch_size=32)
    
    # Task 1: Dataset Overview
    classifier.load_and_explore_data()
    
    # Task 3: Create data generators with augmentation (Task 5)
    classifier.create_data_generators(use_augmentation=True)
    
    # Task 2: Build CNN model
    classifier.build_cnn_model()
    
    # Task 3: Compile model (Task 5: Experiment with hyperparameters)
    classifier.compile_model(learning_rate=0.001)
    
    # Task 3: Train model
    classifier.train_model(epochs=5)
    
    # Task 4: Plot training history
    classifier.plot_training_history()
    
    # Task 4: Evaluate model
    classifier.evaluate_model()
    
    # Additional: Show sample predictions
    classifier.predict_samples(num_samples=10)
    
    # Save the trained model
    classifier.save_model('intel_image_classifier_final.h5')
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  - dataset_samples.png")
    print("  - training_history.png")
    print("  - confusion_matrix.png")
    print("  - sample_predictions.png")
    print("  - best_model.h5")
    print("  - intel_image_classifier_final.h5")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
