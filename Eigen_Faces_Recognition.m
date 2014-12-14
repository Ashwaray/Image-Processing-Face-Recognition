clc;
persons = 40;
training_samples_per_person = 6;
testing_samples_per_person = 4;
total_samples_per_person = 10;
filename = 'C:\Users\ROBITA\Desktop\att_faces\s';
row = 112;
col = 92;

face_database = zeros((row*col),(persons*training_samples_per_person));
mean_face = zeros(row*col,1);
mean_aligned_face_database = zeros((row*col),(persons*training_samples_per_person));
test_image = zeros(row*col,1);
mean_aligned_test_image = zeros(row*col,1);
n = 1;
c = 1;

for i = 1:persons
    
    person_path = strcat(filename,num2str(i),'\');
    
    for j = 1:training_samples_per_person
        
        final_path = strcat(person_path,num2str(j),'.pgm');
        img = imread(final_path,'pgm');
        figure(1),imshow(img),title(i),impixelinfo()
        
        for p = 1:row
            
            for k = 1:col
                
                face_database(c,n) = img(p,k);
                c = c+1;
            end
        end
        n = n + 1;
        c = 1;
    end
end

mean_face = mean(face_database,2);

for i = 1:(persons*training_samples_per_person)
    
    mean_aligned_face_database(:,i) = face_database(:,i)-mean_face;
end

covariance_face = cov(mean_aligned_face_database);
[eigen_vectors,eigen_values] = eig(covariance_face);
eigen_values = eig(covariance_face);
denominator = sum(eigen_values.^2);
optimal_eigen_array = zeros(size(eigen_values));

for i = 1:size(eigen_values)
    
    numerator = 0;
    
    for j = 1:i
        
        numerator = numerator + eigen_values(j).^2;
    end
    optimal_eigen_array(i) = 100*(numerator/denominator);
end
plot(1:size(eigen_values),optimal_eigen_array,'r*-.');

%for optimal_k = 1:(persons*training_samples_per_person)

accuracy_count = 0;
principal_eigen_vectors = eigen_vectors(:,211:240);
projected_face_database = principal_eigen_vectors'*mean_aligned_face_database';
face_signature_database = projected_face_database*mean_aligned_face_database;

for i = 1:persons
    
    person_path = strcat(filename,num2str(i),'\');
    
    for j = 7:10
        
        final_path = strcat(person_path,num2str(j),'.pgm');
        img = imread(final_path,'pgm');
        figure(1),imshow(img),title(i),impixelinfo()
        c = 1;
        
        for p = 1:row
            
            for k = 1:col
                
                test_image(c,1) = img(p,k);
                c = c+1;
                
            end
        end
        
        mean_aligned_test_image = test_image-mean_face;
        projected_test_face_signature = projected_face_database*mean_aligned_test_image;
        temp_distance = flintmax;
        class_label = 9999;
        
        for h = 1:(persons*training_samples_per_person)
            
            distance = 0.5*(sum((face_signature_database(:,h)-projected_test_face_signature).^2));
            
            if distance < temp_distance
                
                temp_distance = distance;
                class_label = h;
            end
        end
        i;
        class_label;
        
        final_label = ceil(class_label/training_samples_per_person);
        
        if final_label == i
            
            accuracy_count = accuracy_count + 1;
            final_path = strcat(person_path,'1','.pgm');
            img = imread(final_path,'pgm');
            figure(3),imshow(img),title(i),impixelinfo()
        else
            i
        end
    end
end
fprintf('Total Accuracy Count is : %f\n', (accuracy_count/160)*100);
% accuracy_plot(optimal_k) = ((accuracy_count/160)*100);
% end
% plot(x,accuracy_plot,'r*-');

