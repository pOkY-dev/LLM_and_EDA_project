from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255, blank=True)
    content = models.TextField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title

    class Meta:
        ordering = ('title',)

class Note(models.Model):
    title = models.CharField(max_length=100, blank=True)
    content = models.TextField()

class Runnable:
    def run(self, input_data):
        pass

class RobertaWrapper(Runnable):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def run(self, input_data):
        # This method is required by the Runnable interface
        # Implement any logic here to process input_data using the tokenizer and model
        pass
