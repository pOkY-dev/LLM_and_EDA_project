from django.shortcuts import render, redirect
from .forms import SearchForm
from .models import Document, Note, RobertaWrapper
from django.db.models import Q

from sentence_transformers import SentenceTransformer, util
import torch

def editor(request):
    docid = int(request.GET.get('docid', 0))
    documents = Document.objects.all()

    if request.method == 'POST':
        docid = int(request.POST.get('docid', 0))
        title = request.POST.get('title')
        content = request.POST.get('content', '')

        if docid > 0:
            document = Document.objects.get(pk=docid)
            document.title = title
            document.content = content
            document.save()

            return redirect('/?docid=%i' % docid)
        else:
            document = Document.objects.create(title=title, content=content)

            return redirect('/?docid=%i' % document.id)

    if docid > 0:
        document = Document.objects.get(pk=docid)
    else:
        document = ''

    context = {
        'docid': docid,
        'documents': documents,
        'document': document
    }

    return render(request, 'editor.html', context)

def delete_document(request, docid):
    document = Document.objects.get(pk=docid)
    document.delete()

    return redirect('/?docid=0')

def search_documents(request):
    query = request.GET.get('query', '')
    documents = Document.objects.all()
    return render(request, 'editor.html', {'documents': documents, 'query': query})


sent_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

def retrieve_from_db(query: str):
    # Use Django's ORM to query the database
    all_documents = Document.objects.all()

    # Extract document titles
    document_titles = [doc.title for doc in all_documents]

    # Compute embeddings for titles and query
    title_embeddings = sent_model.encode(document_titles, convert_to_tensor=True)
    query_embedding = sent_model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query and the titles
    cosine_similarities = util.pytorch_cos_sim(query_embedding, title_embeddings)[0]

    # Sort the documents by their cosine similarity and keep the top 5
    top_indices = torch.topk(cosine_similarities, k=5).indices.tolist()  # Convert tensor to list
    top_documents = [(all_documents[i].content, all_documents[i].title) for i in top_indices]

    return top_documents

def ask_question(request):
    if request.method == 'POST':
        question = request.POST.get('question', '')

        if question:
            # Retrieve relevant documents and titles from the database
            top_documents = retrieve_from_db(question)

            if not top_documents:
                return render(request, 'search_results.html', {'question': question, 'answer': "Sorry, but there are no suitable notes for your question."})

            # Select the document with the highest similarity score
            most_similar_document, most_similar_title = max(top_documents, key=lambda x: util.pytorch_cos_sim(sent_model.encode(x[1], convert_to_tensor=True), sent_model.encode(question, convert_to_tensor=True)))

            # Return the entire context of the most similar document
            answer = most_similar_document

            return render(request, 'search_results.html', {'question': question, 'answer': answer})

    return render(request, 'search_results.html')      