clc;
persons = 40;
training_samples_per_person = 6;
testing_samples_per_person = 4;
total_samples_per_person = 10;
filename = 'C:\Users\ROBITA\Desktop\att_faces\s';
row = 112;
col = 92;

face_database = zeros((row*col),(training_samples_per_person*persons));
mean_face = zeros(row*col,1);
test_image = zeros(row*col,1);
mean_aligned_face_database = zeros((row*col),(training_samples_per_person*persons));
surrogate_covariance = zeros((training_samples_per_person*persons),(training_samples_per_person*persons));
k_best_features_pca = 211;
mean_projected_faces = zeros(30,1);
mean_of_each_person = zeros(30,40);
within_class_scatter_matrix = zeros(30,30);
between_class_scatter_matrix = zeros(30,30);
optimized_objective_function = zeros(30,30);

n = 1;
c = 1;

for i = 1:persons
    
    file_person = strcat(filename,num2str(i),'\');
    
    for j = 1:training_samples_per_person
        
        image_file = strcat(file_person,num2str(j),'.pgm');
        img = imread(image_file,'pgm');
        figure(1),imshow(img),title(i),impixelinfo()
        c = 1;
        for p = 1:row
            
            for k = 1:col
                
                face_database(c,n) = img(p,k);
                c = c + 1;
            end
        end
        
        n = n + 1;
    end
end

mean_face = mean(face_database,2);

for i = 1:persons*training_samples_per_person
    
    mean_aligned_face_database(:,i) = face_database(:,i)-mean_face;
end

surrogate_covariance = cov(mean_aligned_face_database);
[eigen_vectors_pca,eigen_values_pca] = eig(surrogate_covariance);
eigen_values_pca = eig(surrogate_covariance);
projected_face_pca = eigen_vectors_pca(:,k_best_features_pca:(training_samples_per_person*persons))'*mean_aligned_face_database';
face_signature_pca = projected_face_pca*mean_aligned_face_database;

mean_projected_faces = mean(face_signature_pca,2);
first_index = 1;
next_index = training_samples_per_person;

for i = 1:persons
    
    mean_of_each_person(:,i) = mean(face_signature_pca(:,first_index:next_index),2);
    first_index = first_index + training_samples_per_person;
    next_index = next_index + training_samples_per_person;
    
end

first_index = 1;
next_index = training_samples_per_person;

for i = 1:persons
    
    within_class_scatter_matrix = within_class_scatter_matrix + cov((face_signature_pca(:,first_index:next_index))');
    first_index = first_index + training_samples_per_person;
    next_index = next_index + training_samples_per_person;
    
end

for i = 1:persons
    
    between_class_scatter_matrix = between_class_scatter_matrix + (mean_projected_faces-mean_of_each_person(:,i))*(mean_projected_faces-mean_of_each_person(:,i))';
end

between_class_scatter_matrix = 6.*between_class_scatter_matrix;
inverse_within_class_scatter_matrix = inv(within_class_scatter_matrix);
optimized_objective_function = inverse_within_class_scatter_matrix * between_class_scatter_matrix;
[eigen_vectors_lda,eigen_values_lda] = eig(optimized_objective_function);
eigen_values_lda = eig(optimized_objective_function);
accuracy_count_array = zeros(1,30);
best_w = 19;
%for best_w = 1:30
    
    weighted_feature_vector = eigen_vectors_lda(:,1:best_w);
    fisher_faces_databases = weighted_feature_vector'*face_signature_pca;
    accuracy_count = 0;
    optimal_choice_eigen_vectors = zeros(size(eigen_values_lda));
    denominator = sum(eigen_values_lda.^2);
    
    for i = 1:size(eigen_values_lda) 
        
        numerator = 0;
        
        for j = 1:i
            
            numerator = numerator + eigen_values_lda(j)^2;
        end 
        
        optimal_choice_eigen_vectors(i) = 100*(numerator/denominator); 
    end
            
    plot(1:size(eigen_values_lda),optimal_choice_eigen_vectors,'b*-');        
            
    for i = 1:persons
        
        file_person = strcat(filename,num2str(i),'\');
        
        for j = 7:10
            
            image_file = strcat(file_person,num2str(j),'.pgm');
            img = imread(image_file,'pgm');
            figure(1),imshow(img),title(i),impixelinfo()
            c = 1;
            
            for p = 1:row
                
                for k = 1:col
                    
                    test_image(c,1) = img(p,k);
                    c = c+1;
                end
            end
            
            mean_aligned_test_image = test_image - mean_face;
            test_eigen_face = projected_face_pca*mean_aligned_test_image;
            test_fisher_face = weighted_feature_vector'*test_eigen_face;
            temp_distance = flintmax;
            class_label = 999;
            distance = -999;
            
            for h = 1:(persons*training_samples_per_person)
                
                distance = sqrt(sum((fisher_faces_databases(:,h)-test_fisher_face).^2));
                
                if distance < temp_distance
                    
                    temp_distance = distance;
                    class_label = h;
                end
            end
           
            class_label = ceil(class_label/training_samples_per_person);
            if class_label == i
                
                accuracy_count = accuracy_count + 1;
                matching_image = imread(strcat(filename,num2str(i),'\','1.pgm'),'pgm');
                figure(2),title(i),imshow(matching_image),impixelinfo() 
            else
                i
            end
        end
    end 
    
    fprintf('The overall accuracy is %f \n',(accuracy_count/160)*100);
%     accuracy_count_array(best_w) = (accuracy_count/160)*100; 
    
%end
% x = 1:30;
% plot(x,accuracy_count_array,'g*-');



